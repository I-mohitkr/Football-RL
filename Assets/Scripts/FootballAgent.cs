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
///   5. Behavior Parameters → Space Size = 14, Continuous Actions = 3.
///   6. Optionally add a RayPerceptionSensor3D for richer observations.
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
    public float moveSpeed = 0.8f;

    [Tooltip("How fast the agent rotates left/right.")]
    public float turnSpeed = 100f;

    [Header("=== Kick Settings ===")]
    [Tooltip("Force applied to the ball when the agent kicks it.")]
    public float kickForce = 1.5f;

    [Tooltip("Maximum distance from the ball to register a kick.")]
    public float kickRange = 0.3f;

    // ─── Private Fields ──────────────────────────────────────────────
    private Rigidbody agentRb;
    private Vector3 agentStartPos;
    private Quaternion agentStartRot;
    private Vector3 ballStartPos;

    private float existentialRewardPerStep;
    private int stepsSinceLastTouch;
    private bool touchedBallThisEpisode;

    // ─── Constants ───────────────────────────────────────────────────
    private const float GOAL_REWARD         =  1.0f;
    private const float CONCEDE_PENALTY     = -1.0f;
    private const float BALL_TOUCH_REWARD   =  0.05f;
    private const float APPROACH_REWARD     =  0.001f;
    private const float KICK_TOWARD_GOAL    =  0.1f;
    private const float IDLE_PENALTY_RATE   = -0.0005f;
    private const float OUT_OF_BOUNDS_PEN   = -0.5f;
    private const int   MAX_IDLE_STEPS      =  500;

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
        ballStartPos  = ballTransform.localPosition;

        // Small negative reward per step to encourage fast play
        // Guard against MaxStep = 0 (which would cause -Infinity)
        existentialRewardPerStep = MaxStep > 0 ? -1f / MaxStep : -0.0001f;
    }

    // ═════════════════════════════════════════════════════════════════
    //  EPISODE BEGIN — Reset positions & velocities
    // ═════════════════════════════════════════════════════════════════

    public override void OnEpisodeBegin()
    {
        // Penalize if the agent never touched the ball last episode
        if (!touchedBallThisEpisode && StepCount > 0)
        {
            AddReward(-0.2f);
        }

        // Reset agent
        if (agentRb == null) agentRb = GetComponent<Rigidbody>();
        
        transform.localPosition = agentStartPos;
        transform.localRotation = agentStartRot;
        
        if (agentRb != null)
        {
            agentRb.linearVelocity  = Vector3.zero;
            agentRb.angularVelocity = Vector3.zero;
        }

        // Reset ball (only one agent should reset the ball to avoid double-reset)
        if (team == Team.Blue)
        {
            ResetBall();
        }

        // Reset tracking
        stepsSinceLastTouch    = 0;
        touchedBallThisEpisode = false;
    }

    private void ResetBall()
    {
        ballTransform.localPosition = ballStartPos;
        ballRb.linearVelocity       = Vector3.zero;
        ballRb.angularVelocity      = Vector3.zero;

        // Give the ball a tiny random nudge so episodes aren't identical
        // (Commented out because user wants the ball perfectly still at the start)
        // Vector3 randomNudge = new Vector3(
        //     Random.Range(-0.5f, 0.5f),
        //     0f,
        //     Random.Range(-0.5f, 0.5f)
        // );
        // ballRb.AddForce(randomNudge, ForceMode.VelocityChange);
    }

    private void FixedUpdate()
    {
        // Safely end episode in the physics loop to prevent ML-Agents crashes
        if (wantsToEndEpisode)
        {
            wantsToEndEpisode = false;
            EndEpisode();
            return;
        }
    }

    // ═════════════════════════════════════════════════════════════════
    //  OBSERVATIONS — 14 floats total
    // ═════════════════════════════════════════════════════════════════

    public override void CollectObservations(VectorSensor sensor)
    {
        // --- Agent's own state (5 floats) ---
        // Normalized local position (x, z)
        Vector3 agentLocalPos = transform.localPosition;
        sensor.AddObservation(agentLocalPos.x);                     // 1
        sensor.AddObservation(agentLocalPos.z);                     // 2

        // Agent's forward direction (x, z)
        sensor.AddObservation(transform.forward.x);                 // 3
        sensor.AddObservation(transform.forward.z);                 // 4

        // Agent's speed
        sensor.AddObservation(agentRb.linearVelocity.magnitude);    // 5

        // --- Ball state (5 floats) ---
        // Relative position of ball to agent
        Vector3 ballRelative = ballTransform.localPosition - agentLocalPos;
        sensor.AddObservation(ballRelative.x);                      // 6
        sensor.AddObservation(ballRelative.y);                      // 7
        sensor.AddObservation(ballRelative.z);                      // 8

        // Ball velocity (x, z)
        sensor.AddObservation(ballRb.linearVelocity.x);             // 9
        sensor.AddObservation(ballRb.linearVelocity.z);             // 10

        // --- Goal directions (4 floats) ---
        // Direction to opponent goal (attack target)
        Vector3 toOpponentGoal = (opponentGoal.localPosition - agentLocalPos).normalized;
        sensor.AddObservation(toOpponentGoal.x);                    // 11
        sensor.AddObservation(toOpponentGoal.z);                    // 12

        // Direction to own goal (defend target)
        Vector3 toOwnGoal = (ownGoal.localPosition - agentLocalPos).normalized;
        sensor.AddObservation(toOwnGoal.x);                        // 13
        sensor.AddObservation(toOwnGoal.z);                        // 14
    }

    // ═════════════════════════════════════════════════════════════════
    //  ACTIONS — 3 continuous actions
    // ═════════════════════════════════════════════════════════════════

    public override void OnActionReceived(ActionBuffers actions)
    {
        // --- Parse actions ---
        float moveZ    = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f); // forward/backward (W/S)
        float moveX    = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f); // left/right (A/D)
        float kickInput = Mathf.Clamp(actions.ContinuousActions[2],  0f, 1f); // kick power (0-1)

        // --- Move (world-space WASD movement) ---
        Vector3 move = new Vector3(moveX, 0f, moveZ);
        // Clamp diagonal movement so it's not faster than straight movement
        if (move.magnitude > 1f) move.Normalize();
        Vector3 desiredVelocity = move * moveSpeed;
        Vector3 currentVel = agentRb.linearVelocity;
        // Only control horizontal movement, preserve vertical velocity (gravity)
        agentRb.linearVelocity = new Vector3(desiredVelocity.x, currentVel.y, desiredVelocity.z);

        // --- Face movement direction ---
        if (move.magnitude > 0.1f)
        {
            transform.rotation = Quaternion.LookRotation(move);
        }

        // --- Kick ---
        if (kickInput > 0.5f)
        {
            TryKickBall();
        }

        // --- Per-step rewards ---
        // Small existential penalty to encourage fast play
        AddReward(existentialRewardPerStep);

        // Reward for approaching the ball (curriculum shaping)
        float distToBall = Vector3.Distance(transform.localPosition, ballTransform.localPosition);
        AddReward(APPROACH_REWARD / (1f + distToBall));

        // Idle penalty — if agent hasn't touched the ball in too many steps
        stepsSinceLastTouch++;
        if (stepsSinceLastTouch > MAX_IDLE_STEPS)
        {
            AddReward(IDLE_PENALTY_RATE);
        }
    }

    private void TryKickBall()
    {
        float distToBall = Vector3.Distance(transform.position, ballTransform.position);

        if (distToBall <= kickRange)
        {
            // Direction from agent to ball
            Vector3 kickDir = (ballTransform.position - transform.position).normalized;

            // Blend the kick direction toward the opponent goal for smarter kicks
            Vector3 toGoal  = (opponentGoal.position - ballTransform.position).normalized;
            Vector3 finalKickDir = (kickDir + toGoal * 0.5f).normalized;

            ballRb.AddForce(finalKickDir * kickForce, ForceMode.VelocityChange);

            // Reward for touching the ball
            AddReward(BALL_TOUCH_REWARD);
            stepsSinceLastTouch    = 0;
            touchedBallThisEpisode = true;

            // Bonus reward if the kick is aimed toward the opponent's goal
            float dotToGoal = Vector3.Dot(kickDir, toGoal);
            if (dotToGoal > 0.5f)
            {
                AddReward(KICK_TOWARD_GOAL * dotToGoal);
            }
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
    }

    // ═════════════════════════════════════════════════════════════════
    //  COLLISION & TRIGGER DETECTION
    // ═════════════════════════════════════════════════════════════════

    private void OnCollisionEnter(Collision collision)
    {
        // Reward for physically touching the ball (body contact, not just kick)
        if (collision.gameObject.CompareTag("Ball"))
        {
            AddReward(BALL_TOUCH_REWARD * 0.5f);
            stepsSinceLastTouch    = 0;
            touchedBallThisEpisode = true;
        }
    }

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
            // This agent conceded!
            AddReward(CONCEDE_PENALTY);
        }

        // Queue episode end safely for the next physics step
        wantsToEndEpisode = true;
    }

    /// <summary>
    /// Call this if the ball or agent goes out of bounds.
    /// </summary>
    public void OutOfBounds()
    {
        AddReward(OUT_OF_BOUNDS_PEN);
        EndEpisode();
        if (opponent != null)
        {
            opponent.EndEpisode();
        }
    }
}
