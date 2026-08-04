using UnityEngine;
using UnityEngine.InputSystem;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

/// <summary>
/// FootballAgent — ML-Agents 4.x agent for a 1v1 Messi vs Ronaldo football game.
///
/// Setup (per agent capsule in the Inspector):
///   1. Assign the Ball, Own Goal, and Opponent Goal transforms.
///   2. Assign the Ground transform (used for resetting positions).
///   3. Tag your goal trigger zones: "BlueGoal" and "RedGoal".
///   4. Set Team to Blue for one agent and Red for the other.
///   5. Behavior Parameters → Space Size = 24, Continuous Actions = 4.
///      Observation breakdown:
///        6  agent state      (pos.x, pos.z, fwd.x, fwd.z, vel.x, vel.z)
///        1  agent angular vel (yaw rate)
///        6  ball state       (rel.x, rel.y, rel.z, dist, velLocal.x, velLocal.z)
///        1  possession flag  (ball within kick range)
///        4  goal directions  (toOpponentGoal.x/z, toOwnGoal.x/z)
///        4  opponent state   (rel.x, rel.z, velLocal.x, velLocal.z)
///        2  opponent facing  (fwd.x, fwd.z, local space)
///        -----------------------------------------------------------
///        24 total
///      Action breakdown: moveZ, moveX, kick trigger, kick power = 4
///   6. Optionally add a RayPerceptionSensor3D for richer observations.
///
/// Design notes (intentional, confirmed choices — do not "fix" these):
///   - CONCEDE_PENALTY is 0: this is an offense-focused training setup.
///     Neither agent is directly reinforced for defending. A curriculum
///     stage can temporarily override this via the "stage_concede_penalty"
///     environment parameter without changing the final production value.
///   - Kicks use an aim-assist blend toward the opponent's goal (see
///     TryKickBall). Agents do not have to learn precise aim from scratch.
///
/// Curriculum / long-training hooks:
///   This script reads the following Environment Parameters at the start
///   of every episode (set via your trainer YAML's environment_parameters
///   block, or via SideChannels for manual testing). All default to
///   values that reproduce full, non-curriculum gameplay if unset, so the
///   script is safe to use with no curriculum config at all.
///     - "shaping_scale"          (default 1.0)  multiplies ALL shaping
///       rewards (facing/approach/progress/kick). Never affects
///       GOAL_REWARD or OUT_OF_BOUNDS_PEN. Anneal this down over training
///       once the agent can reliably score, so shaping stops acting as a
///       skill ceiling.
///     - "stage_disable_opponent" (default 0.0)  if > 0.5, the opponent
///       capsule is disabled (kinematic + hidden) for empty-net practice.
///       Toggle this from a shared GameManager/curriculum controller, not
///       required for this script to function without one.
///     - "stage_concede_penalty"  (default 0.0)  temporary, curriculum-only
///       override for CONCEDE_PENALTY. Leave at 0 for the final production
///       reward function; set negative only during a dedicated
///       "Stage 5: Defend" curriculum phase.
///     - "stage_goal_distance"    (default 1.0)  scales how far from the
///       agent's own start position the ball spawns, for progressively
///       harder "kick toward goal" practice (0.3 = close, 1.0 = full field).
/// </summary>
public class FootballAgent : Agent
{
    // ─── Inspector Fields ────────────────────────────────────────────
    [Header("=== References ===")]
    [Tooltip("The football Rigidbody.")]
    public Rigidbody ballRb;

    [Tooltip("Transform of the ball.")]
    public Transform ballTransform;

    [Tooltip("Transform of THIS agent's own goal (the one it defends).")]
    public Transform ownGoal;

    [Tooltip("Transform of the OPPONENT's goal (the one it attacks).")]
    public Transform opponentGoal;

    [Tooltip("Transform of the ground plane (used to calculate reset bounds).")]
    public Transform groundTransform;

    [Tooltip("Reference to the opponent FootballAgent (for resetting both).")]
    public FootballAgent opponent;

    [Header("=== Team ===")]
    public Team team = Team.Blue;
    public enum Team { Blue, Red }

    [Header("=== Movement Settings ===")]
    [Tooltip("How fast the agent moves forward/backward.")]
    public float moveSpeed = 4f;

    [Tooltip("How fast the agent rotates left/right.")]
    public float turnSpeed = 100f;

    [Header("=== Kick Settings ===")]
    [Tooltip("Maximum force applied to the ball on a full-power kick.")]
    public float kickForce = 10f;

    [Tooltip("Minimum force applied to the ball on the lightest registered kick (for passes/taps).")]
    public float minKickForce = 2f;

    [Tooltip("Maximum distance from the ball to register a kick.")]
    public float kickRange = 1.5f;

    // ─── Private Fields ──────────────────────────────────────────────
    private Rigidbody agentRb;
    private Rigidbody opponentRb; // cached once instead of GetComponent() every observation call
    private Vector3 agentStartPos;
    private Quaternion agentStartRot;
    private Vector3 ballStartPos;

    private float lastDistToBall;
    private float lastBallDistToGoal;
    private float totalApproachReward = 0f;
    private float totalProgressReward = 0f;
    private float totalFacingReward = 0f;
    private float totalKickReward = 0f;
    private float kickCooldownTimer = 0f;
    private Vector3 currentMoveDir = Vector3.zero;
    private Vector3 cachedDirToBall = Vector3.forward; // computed once per step, reused for kicks

    // Per-pitch (not process-wide) reset synchronization, anchored on the
    // Team.Blue instance of each agent pair purely as a stable, arbitrary
    // place to store the shared counter — NOT a process-wide static, so
    // parallel training environments (--num-envs > 1) don't cross-sync
    // ball resets across unrelated pitches.
    private int pairedGeneration = 0;
    private int localEpisodeGeneration = 0;

    // Curriculum / environment-parameter driven values, re-read every episode
    private float shapingScale = 1f;
    private float stageConcedePenalty = 0f;
    private float stageGoalDistanceScale = 1f;
    private bool stageDisableOpponent = false;

    // ─── Constants ───────────────────────────────────────────────────
    private const float GOAL_REWARD         =  5.0f; // Terminal reward — must stay strictly above the shaping ceiling
    private const float CONCEDE_PENALTY     =  0.0f; // Intentional: offense-focused production training, see class doc
    private const float BALL_TOUCH_REWARD   =  0.1f;
    private const float KICK_TOWARD_GOAL    =  0.5f;
    
    // Increased penalty to -2.0 to fix the "suicide exploit".
    // 10,000 steps of existential penalty = -1.0.
    // The agent realized -0.5 (out of bounds) is BETTER than -1.0 (waiting for timeout).
    // Now, out of bounds is the worst possible outcome!
    private const float OUT_OF_BOUNDS_PEN   = -2.0f; 
    private const float EXISTENTIAL_PENALTY = -0.0001f; // tiny per-step cost to discourage stalling/passivity

    // Rebalanced per-episode shaping caps. Total ceiling = 0.3 + 0.8 + 1.0 + 1.0 = 3.1,
    // comfortably under GOAL_REWARD (5.0) so terminal reward always dominates
    // even in the worst case where an agent maxes out every shaping category
    // without ever scoring.
    private const float MAX_FACING_REWARD_PER_EPISODE   = 0.3f;
    private const float MAX_APPROACH_REWARD_PER_EPISODE = 0.8f;
    private const float MAX_PROGRESS_REWARD_PER_EPISODE = 1.0f;
    private const float MAX_KICK_REWARD_PER_EPISODE     = 1.0f;

    // Per-step clamps so a single physics spike (teleport, bounce, fast
    // dash) can't dump a large fraction of the episode's shaping budget
    // into one step, which would inflate advantage-estimate variance.
    private const float MAX_DIST_DELTA_PER_STEP = 1f;

    // Fix for ML-Agents Coroutine crashes: safely queue the episode end
    private bool wantsToEndEpisode = false;

    // ═════════════════════════════════════════════════════════════════
    //  INITIALIZATION
    // ═════════════════════════════════════════════════════════════════

    public override void Initialize()
    {
        agentRb = GetComponent<Rigidbody>();

        // Freeze rotation so the capsule never tips over
        agentRb.constraints = RigidbodyConstraints.FreezeRotationX
                            | RigidbodyConstraints.FreezeRotationZ;

        // Cache starting positions for episode resets
        agentStartPos = transform.localPosition;
        agentStartRot = transform.localRotation;
        if (ballTransform != null)
        {
            ballStartPos = ballTransform.localPosition;
        }

        // Cache opponent Rigidbody once — avoids a GetComponent() call on
        // every single CollectObservations() invocation for the entire run.
        if (opponent != null)
        {
            opponentRb = opponent.GetComponent<Rigidbody>();
        }
    }

    // ═════════════════════════════════════════════════════════════════
    //  EPISODE BEGIN — Reset positions & velocities
    // ═════════════════════════════════════════════════════════════════

    public override void OnEpisodeBegin()
    {
        ReadCurriculumParameters();

        // Always reset this agent so it doesn't get permanently stuck in
        // physics glitches, regardless of who resets the ball.
        if (agentRb == null) agentRb = GetComponent<Rigidbody>();

        // Remove jitter: If the jitter accidentally placed the agent or ball inside a wall
        // or the ground, Unity's physics engine would violently shoot them out of bounds!
        // We now reset them to the EXACT starting positions.
        transform.localPosition = agentStartPos;
        transform.localRotation = agentStartRot;

        if (agentRb != null)
        {
            agentRb.linearVelocity  = Vector3.zero;
            agentRb.angularVelocity = Vector3.zero;
        }

        // Shared ball reset, scoped to this agent/opponent pair only.
        // If local == synced, it means this OnEpisodeBegin was triggered by a Max Step
        // timeout, NOT a goal (which bumps the counter before EndEpisode). 
        // We must manually bump the generation so the ball resets!
        if (localEpisodeGeneration == GetSyncedGeneration())
        {
            SetSyncedGeneration(GetSyncedGeneration() + 1);
        }

        if (localEpisodeGeneration != GetSyncedGeneration())
        {
            localEpisodeGeneration = GetSyncedGeneration();

            if (ballTransform != null && ballRb != null)
            {
                ballTransform.localPosition = ballStartPos;
                ballRb.linearVelocity  = Vector3.zero;
                ballRb.angularVelocity = Vector3.zero;
            }
        }

        // Curriculum: optionally disable the opponent for empty-net practice
        if (opponent != null)
        {
            opponent.gameObject.SetActive(!stageDisableOpponent);
            if (opponentRb != null) opponentRb.detectCollisions = !stageDisableOpponent;
        }

        // Reset tracking
        lastDistToBall = Vector3.Distance(transform.localPosition, ballTransform.localPosition);
        lastBallDistToGoal = Vector3.Distance(ballTransform.localPosition, opponentGoal.localPosition);
        totalApproachReward = 0f;
        totalProgressReward = 0f;
        totalFacingReward = 0f;
        totalKickReward = 0f;
        kickCooldownTimer = 0f;
    }

    /// <summary>
    /// Reads all curriculum / long-training environment parameters for this
    /// episode. Every parameter has a default that reproduces full,
    /// non-curriculum production gameplay if the trainer config doesn't set
    /// it, so this script works standalone with zero curriculum configuration.
    /// </summary>
    private void ReadCurriculumParameters()
    {
        var ep = Academy.Instance.EnvironmentParameters;
        shapingScale           = ep.GetWithDefault("shaping_scale", 1f);
        stageConcedePenalty    = ep.GetWithDefault("stage_concede_penalty", 0f);
        stageGoalDistanceScale = ep.GetWithDefault("stage_goal_distance", 1f);
        stageDisableOpponent   = ep.GetWithDefault("stage_disable_opponent", 0f) > 0.5f;
    }

    // Minimal cross-instance generation sync without a process-wide static.
    // Anchored on whichever instance is Team.Blue purely as an arbitrary,
    // stable place for the counter to live (not a claim about who resets —
    // reset ownership is still first-come via localEpisodeGeneration above).
    private int GetSyncedGeneration()
    {
        FootballAgent owner = (team == Team.Blue) ? this : opponent;
        return owner != null ? owner.pairedGeneration : pairedGeneration;
    }

    private void SetSyncedGeneration(int value)
    {
        FootballAgent owner = (team == Team.Blue) ? this : opponent;
        if (owner != null) owner.pairedGeneration = value;
        else pairedGeneration = value;
    }

    private void FixedUpdate()
    {
        if (kickCooldownTimer > 0f)
        {
            kickCooldownTimer -= Time.fixedDeltaTime;
        }

        // Safely end episode in the physics loop to prevent ML-Agents crashes.
        // Both GoalScored() and OutOfBounds() route through this same queue,
        // so neither path can call EndEpisode() mid-callback.
        if (wantsToEndEpisode)
        {
            wantsToEndEpisode = false;
            EndEpisode();
            return;
        }

        // Apply smooth movement every physics frame
        if (agentRb != null)
        {
            Vector3 desiredVelocity = currentMoveDir * moveSpeed;
            Vector3 targetVelocity = new Vector3(desiredVelocity.x, agentRb.linearVelocity.y, desiredVelocity.z);
            agentRb.linearVelocity = Vector3.Lerp(agentRb.linearVelocity, targetVelocity, Time.fixedDeltaTime * 10f);

            if (currentMoveDir.magnitude > 0.1f)
            {
                Quaternion targetRotation = Quaternion.LookRotation(currentMoveDir);
                transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, Time.fixedDeltaTime * 15f);
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════
    //  OBSERVATIONS — 24 floats total (see Space Size note in class doc)
    // ═════════════════════════════════════════════════════════════════

    public override void CollectObservations(VectorSensor sensor)
    {
        // --- Agent's own state (6 floats: position + facing + velocity X/Z) ---
        Vector3 agentLocalPos = transform.localPosition;
        sensor.AddObservation(agentLocalPos.x / 20f);                // 1
        sensor.AddObservation(agentLocalPos.z / 10f);                // 2
        sensor.AddObservation(transform.forward.x);                  // 3
        sensor.AddObservation(transform.forward.z);                  // 4

        // Signed velocity components (local space) instead of scalar speed —
        // magnitude alone can't distinguish "toward ball" from "away from
        // ball" at the same speed, which forces the value function to
        // disambiguate states it shouldn't need to.
        Vector3 agentVelLocal = transform.InverseTransformDirection(agentRb.linearVelocity);
        sensor.AddObservation(agentVelLocal.x / 10f);                 // 5
        sensor.AddObservation(agentVelLocal.z / 10f);                 // 6

        // --- Agent angular velocity (1 float) ---
        // Lets the agent anticipate turn overshoot given the Slerp-based
        // rotation in FixedUpdate, reducing oscillation near the ball.
        sensor.AddObservation(agentRb.angularVelocity.y / 10f);       // 7

        // --- Ball state (6 floats) ---
        Vector3 ballLocal = transform.InverseTransformPoint(ballTransform.position);
        sensor.AddObservation(ballLocal.x / 20f);                     // 8
        sensor.AddObservation(ballLocal.y / 5f);                      // 9
        sensor.AddObservation(ballLocal.z / 10f);                     // 10

        float distToBall = Vector3.Distance(transform.position, ballTransform.position);
        sensor.AddObservation(distToBall / 20f);                      // 11

        Vector3 ballVelLocal = transform.InverseTransformDirection(ballRb.linearVelocity);
        sensor.AddObservation(ballVelLocal.x / 10f);                  // 12
        sensor.AddObservation(ballVelLocal.z / 10f);                  // 13

        // --- Possession flag (1 float) ---
        // Clean, low-noise signal for "is the ball in my control radius
        // right now" — supports learning dribble-vs-chase behavior without
        // re-deriving possession from raw distance every time.
        sensor.AddObservation(distToBall <= kickRange ? 1f : 0f);     // 14

        // --- Goal directions (4 floats) ---
        Vector3 toOpponentGoal = transform.InverseTransformPoint(opponentGoal.position).normalized;
        sensor.AddObservation(toOpponentGoal.x);                      // 15
        sensor.AddObservation(toOpponentGoal.z);                      // 16

        Vector3 toOwnGoal = transform.InverseTransformPoint(ownGoal.position).normalized;
        sensor.AddObservation(toOwnGoal.x);                           // 17
        sensor.AddObservation(toOwnGoal.z);                           // 18

        // --- Opponent state (4 floats) + facing (2 floats) ---
        if (opponent != null && opponentRb != null && opponent.gameObject.activeInHierarchy)
        {
            Vector3 oppLocal = transform.InverseTransformPoint(opponent.transform.position);
            sensor.AddObservation(oppLocal.x / 20f);                  // 19
            sensor.AddObservation(oppLocal.z / 10f);                  // 20

            Vector3 oppVelLocal = transform.InverseTransformDirection(opponentRb.linearVelocity);
            sensor.AddObservation(oppVelLocal.x / 10f);               // 21
            sensor.AddObservation(oppVelLocal.z / 10f);               // 22

            // Opponent facing direction (local space) — first-order signal
            // for anticipating tackles, blocking shots, or exploiting space.
            Vector3 oppForwardLocal = transform.InverseTransformDirection(opponent.transform.forward);
            sensor.AddObservation(oppForwardLocal.x);                 // 23
            sensor.AddObservation(oppForwardLocal.z);                 // 24
        }
        else
        {
            // Fallback if opponent is missing or disabled (curriculum
            // empty-net stage) — still must add exactly 6 floats to keep
            // the total observation count fixed.
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }
    }

    // ═════════════════════════════════════════════════════════════════
    //  ACTIONS — 4 continuous actions
    // ═════════════════════════════════════════════════════════════════

    public override void OnActionReceived(ActionBuffers actions)
    {
        // --- Parse actions ---
        float moveZ     = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f); // forward/backward
        float moveX     = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f); // left/right
        float kickInput = Mathf.Clamp(actions.ContinuousActions[2], 0f, 1f);  // kick trigger (0-1)
        float kickPower = Mathf.Clamp01(actions.ContinuousActions[3]);        // kick power (0=soft pass, 1=full shot)

        // --- Move (Update target direction for FixedUpdate) ---
        currentMoveDir = (transform.forward * moveZ + transform.right * moveX);
        currentMoveDir.y = 0f;
        currentMoveDir = currentMoveDir.normalized;

        // Cache direction-to-ball once per step; reused by TryKickBall so we
        // don't recompute the same normalized vector (and its sqrt) twice
        // in the same step.
        cachedDirToBall = (ballTransform.position - transform.position).normalized;

        // --- Kick ---
        if (kickInput > 0.5f && kickCooldownTimer <= 0f)
        {
            TryKickBall(cachedDirToBall, kickPower);
        }

        // Tiny existential penalty every step to discourage passive
        // stalling policies; negligible relative to any real shaping
        // reward but adds pressure against "do nothing" equilibria.
        AddReward(EXISTENTIAL_PENALTY);

        // --- Facing Reward (capped, gated by proximity) ---
        // Only pays out near the ball — otherwise an agent could farm this
        // safely from a distance without ever engaging, since facing the
        // ball from afar carries zero risk and requires no real skill.
        float facingDot = Vector3.Dot(transform.forward, cachedDirToBall);
        float distToBallNow = Vector3.Distance(transform.position, ballTransform.position);
        if (facingDot > 0.5f
            && distToBallNow < kickRange * 2f
            && totalFacingReward < MAX_FACING_REWARD_PER_EPISODE)
        {
            float facingReward = facingDot * 0.001f * shapingScale;
            AddReward(facingReward);
            totalFacingReward += facingReward;
        }

        // Potential-based approach reward to guide agent to ball
        float currentDistToBall = Vector3.Distance(transform.localPosition, ballTransform.localPosition);
        float currentBallDistToGoal = Vector3.Distance(ballTransform.localPosition, opponentGoal.localPosition);

        // Prevent massive teleportation penalties/rewards on the first frame after a goal resets the ball
        if (StepCount <= 1)
        {
            lastDistToBall = currentDistToBall;
            lastBallDistToGoal = currentBallDistToGoal;
        }

        // 1. Capped, per-step-clamped Approach Reward (Breadcrumbs)
        // Clamping the raw delta prevents a single physics spike (teleport,
        // bounce, fast dash) from dumping a large chunk of the episode's
        // shaping budget into one step, which would otherwise inflate the
        // variance of that step's advantage estimate and destabilize PPO's
        // clipped surrogate objective.
        float approachDelta = Mathf.Clamp(lastDistToBall - currentDistToBall, -MAX_DIST_DELTA_PER_STEP, MAX_DIST_DELTA_PER_STEP);
        float approachReward = approachDelta * 0.1f * shapingScale;
        if (approachReward > 0 && totalApproachReward < MAX_APPROACH_REWARD_PER_EPISODE)
        {
            AddReward(approachReward);
            totalApproachReward += approachReward;
        }
        lastDistToBall = currentDistToBall;

        // 2. Capped, per-step-clamped Ball Progress Reward
        float progressDelta = Mathf.Clamp(lastBallDistToGoal - currentBallDistToGoal, -MAX_DIST_DELTA_PER_STEP, MAX_DIST_DELTA_PER_STEP);
        float ballProgressReward = progressDelta * 0.1f * shapingScale;
        if (ballProgressReward > 0 && totalProgressReward < MAX_PROGRESS_REWARD_PER_EPISODE)
        {
            AddReward(ballProgressReward);
            totalProgressReward += ballProgressReward;
        }
        lastBallDistToGoal = currentBallDistToGoal;
    }

    private void TryKickBall(Vector3 dirToBall, float kickPower)
    {
        float distToBall = Vector3.Distance(transform.position, ballTransform.position);

        // Require rough facing alignment with the ball to register a kick —
        // otherwise the aim-assist blend lets an agent "kick" a ball that's
        // behind it just by being in range, which decouples the facing
        // observation/reward from the action that actually matters.
        float facingAlignment = Vector3.Dot(transform.forward, dirToBall);

        if (distToBall <= kickRange && facingAlignment > 0f)
        {
            // Blend the kick direction toward the opponent goal for smarter kicks.
            // Intentional aim-assist: the agent does not have to learn precise
            // aim purely from raw kick direction.
            Vector3 toGoal = (opponentGoal.position - ballTransform.position).normalized;
            Vector3 finalKickDir = (dirToBall + toGoal * 0.5f).normalized;

            // Variable kick power: blends between minKickForce (soft
            // pass/tap) and kickForce (full shot) based on the 4th action,
            // so the agent can learn WHEN to strike hard vs. control softly
            // instead of every touch being an identical fixed-force kick.
            float appliedForce = Mathf.Lerp(minKickForce, kickForce, kickPower);

            float ballSpeedBefore = ballRb.linearVelocity.magnitude;
            ballRb.AddForce(finalKickDir * appliedForce, ForceMode.VelocityChange);
            float ballSpeedAfter = ballRb.linearVelocity.magnitude;

            // Only reward the kick if it actually changed the ball's motion
            // meaningfully — otherwise an agent can farm BALL_TOUCH_REWARD
            // risk-free by tapping the ball against a wall repeatedly.
            bool kickHadEffect = ballSpeedAfter > ballSpeedBefore + 0.5f;

            if (kickHadEffect && totalKickReward < MAX_KICK_REWARD_PER_EPISODE)
            {
                float touchReward = BALL_TOUCH_REWARD * shapingScale;
                AddReward(touchReward);
                totalKickReward += touchReward;
            }

            // Bonus reward if the kick is aimed toward the opponent's goal
            float dotToGoal = Vector3.Dot(dirToBall, toGoal);
            if (kickHadEffect && dotToGoal > 0.5f && totalKickReward < MAX_KICK_REWARD_PER_EPISODE)
            {
                float bonus = KICK_TOWARD_GOAL * dotToGoal * shapingScale;
                AddReward(bonus);
                totalKickReward += bonus;
            }

            // Set cooldown so they can't spam kick! (1.0 seconds)
            kickCooldownTimer = 1.0f;
        }
    }

    // ═════════════════════════════════════════════════════════════════
    //  HEURISTIC — Manual keyboard control for testing
    // ═════════════════════════════════════════════════════════════════

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuous = actionsOut.ContinuousActions;

        Keyboard kb = Keyboard.current;
        if (kb == null) return;

        // W/S for forward/backward (Z-axis)
        continuous[0] = 0f;
        if (kb.wKey.isPressed || kb.upArrowKey.isPressed)      continuous[0] =  1f;
        if (kb.sKey.isPressed || kb.downArrowKey.isPressed)    continuous[0] = -1f;

        // A/D for left/right strafe (X-axis)
        continuous[1] = 0f;
        if (kb.dKey.isPressed || kb.rightArrowKey.isPressed)   continuous[1] =  1f;
        if (kb.aKey.isPressed || kb.leftArrowKey.isPressed)    continuous[1] = -1f;

        // Space to kick
        continuous[2] = kb.spaceKey.isPressed ? 1f : 0f;

        // Shift+Space for a full-power shot, Space alone for a soft tap —
        // gives manual testers a way to exercise the new kick-power action.
        continuous[3] = kb.leftShiftKey.isPressed ? 1f : 0.3f;
    }

    // ═════════════════════════════════════════════════════════════════
    //  COLLISION & TRIGGER DETECTION
    // ═════════════════════════════════════════════════════════════════

    /// <summary>
    /// Call this from the GoalDetector script when the ball enters a goal zone.
    /// </summary>
    public void GoalScored(Team scoringTeam)
    {
        if (scoringTeam == team)
        {
            // This agent scored!
            AddReward(GOAL_REWARD);
        }
        else
        {
            // This agent conceded. CONCEDE_PENALTY is 0 in production
            // (offense-focused training, confirmed choice). stageConcedePenalty
            // allows a curriculum stage (e.g. "Stage 5: Defend") to
            // temporarily introduce a real penalty without altering the
            // production constant itself.
            AddReward(CONCEDE_PENALTY + stageConcedePenalty);
        }

        // Bump the shared generation BEFORE queuing episode end, so that
        // whichever agent's OnEpisodeBegin runs next (this one or the
        // opponent's) sees a fresh generation number and one of them
        // resets the ball exactly once. Scoped per-pair, not process-wide.
        SetSyncedGeneration(GetSyncedGeneration() + 1);

        // Queue episode end safely for the next physics step
        wantsToEndEpisode = true;
    }

    /// <summary>
    /// Call this if the ball or agent goes out of bounds.
    /// </summary>
    public void OutOfBounds()
    {
        AddReward(OUT_OF_BOUNDS_PEN);

        // Bump generation once here too, for the same reason as GoalScored.
        SetSyncedGeneration(GetSyncedGeneration() + 1);

        // Route through the same safe queue as everything else instead of
        // calling EndEpisode() directly — avoids crashes if invoked from a
        // physics trigger callback mid-FixedUpdate.
        wantsToEndEpisode = true;

        if (opponent != null)
        {
            opponent.QueueEpisodeEnd();
        }
    }

    /// <summary>
    /// Lets another agent (or a shared GoalDetector/GameManager) safely
    /// request that this agent's episode end on the next FixedUpdate,
    /// without calling EndEpisode() directly from a possibly-unsafe context.
    /// </summary>
    public void QueueEpisodeEnd()
    {
        wantsToEndEpisode = true;
    }
}