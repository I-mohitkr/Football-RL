using UnityEngine;
using UnityEngine.InputSystem;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

/// <summary>
/// Streamlined FootballAgent — 100% Offensive Focus.
/// No curriculum, no defensive logic, no aim-assist.
/// The agent must learn to run to the ball and strike it into the goal.
/// </summary>
public class FootballAgent : Agent
{
    [Header("=== References ===")]
    public Rigidbody ballRb;
    public Transform ballTransform;
    public Transform ownGoal;
    public Transform opponentGoal;
    public Transform groundTransform;
    public FootballAgent opponent;

    [Header("=== Team ===")]
    public Team team = Team.Blue;
    public enum Team { Blue, Red }

    [Header("=== Movement Settings ===")]
    public float moveSpeed = 4f;
    public float turnSpeed = 100f;

    [Header("=== Kick Settings ===")]
    public float kickForce = 10f;
    public float kickRange = 1.5f;

    // Private physics caching
    private Rigidbody agentRb;
    private Rigidbody opponentRb;
    private Vector3 agentStartPos;
    private Quaternion agentStartRot;
    private Vector3 ballStartPos;
    private Vector3 currentMoveDir = Vector3.zero;

    // Reward Tracking
    private float minRecordDistToBall;
    private float minRecordBallDistToGoal;
    private bool wantsToEndEpisode = false;
    private bool touchedBallThisEpisode = false;
    private float kickCooldown = 0f;

    // Shared Episode Counter (so only one agent resets the ball)
    private int localEpisodeGeneration = 0;
    private int pairedGeneration = 0;

    // ─── Constants ───────────────────────────────────────────────────
    private const float GOAL_REWARD         =  10.0f; // Doubled to make scoring the ultimate priority
    private const float OUT_OF_BOUNDS_PEN   = -1.0f; 
    
    // REMOVED STEP_PENALTY: It was causing the agent to freeze in place to avoid bleeding points.

    public override void Initialize()
    {
        agentRb = GetComponent<Rigidbody>();
        agentRb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

        agentStartPos = transform.localPosition;
        agentStartRot = transform.localRotation;
        
        if (ballTransform != null) ballStartPos = ballTransform.localPosition;
        if (opponent != null) opponentRb = opponent.GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        // 1. Reset Agent
        if (agentRb == null) agentRb = GetComponent<Rigidbody>();
        transform.localPosition = agentStartPos;
        transform.localRotation = agentStartRot;
        agentRb.linearVelocity = Vector3.zero;
        agentRb.angularVelocity = Vector3.zero;

        // 2. Synchronized Ball Reset
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
                ballRb.linearVelocity = Vector3.zero;
                ballRb.angularVelocity = Vector3.zero;
            }
        }

        // 3. Reset Trackers
        minRecordDistToBall = Vector3.Distance(transform.localPosition, ballTransform.localPosition);
        minRecordBallDistToGoal = Vector3.Distance(ballTransform.localPosition, opponentGoal.localPosition);
        touchedBallThisEpisode = false;
        kickCooldown = 0f;
    }

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
        if (kickCooldown > 0f) kickCooldown -= Time.fixedDeltaTime;

        if (wantsToEndEpisode)
        {
            wantsToEndEpisode = false;
            EndEpisode();
            return;
        }

        // Physics Movement
        if (agentRb != null)
        {
            Vector3 desiredVelocity = currentMoveDir * moveSpeed;
            Vector3 targetVelocity = new Vector3(desiredVelocity.x, agentRb.linearVelocity.y, desiredVelocity.z);
            agentRb.linearVelocity = Vector3.Lerp(agentRb.linearVelocity, targetVelocity, Time.fixedDeltaTime * 10f);

            if (currentMoveDir.magnitude > 0.1f)
            {
                Quaternion targetRot = Quaternion.LookRotation(currentMoveDir);
                transform.rotation = Quaternion.Slerp(transform.rotation, targetRot, Time.fixedDeltaTime * 15f);
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════
    //  OBSERVATIONS (Strictly 24 Floats to match Inspector)
    // ═════════════════════════════════════════════════════════════════
    public override void CollectObservations(VectorSensor sensor)
    {
        // 1-6: Agent State (Pos X/Z, Fwd X/Z, Vel X/Z)
        Vector3 localPos = transform.localPosition;
        sensor.AddObservation(localPos.x / 20f);                // 1
        sensor.AddObservation(localPos.z / 10f);                // 2
        sensor.AddObservation(transform.forward.x);             // 3
        sensor.AddObservation(transform.forward.z);             // 4
        Vector3 velLocal = transform.InverseTransformDirection(agentRb.linearVelocity);
        sensor.AddObservation(velLocal.x / 10f);                // 5
        sensor.AddObservation(velLocal.z / 10f);                // 6

        // 7: Agent Angular Vel
        sensor.AddObservation(agentRb.angularVelocity.y / 10f); // 7

        // 8-13: Ball State
        Vector3 ballLocal = transform.InverseTransformPoint(ballTransform.position);
        sensor.AddObservation(ballLocal.x / 20f);               // 8
        sensor.AddObservation(ballLocal.y / 5f);                // 9
        sensor.AddObservation(ballLocal.z / 10f);               // 10
        float distToBall = Vector3.Distance(transform.position, ballTransform.position);
        sensor.AddObservation(distToBall / 20f);                // 11
        Vector3 ballVelLocal = transform.InverseTransformDirection(ballRb.linearVelocity);
        sensor.AddObservation(ballVelLocal.x / 10f);            // 12
        sensor.AddObservation(ballVelLocal.z / 10f);            // 13

        // 14: Possession Flag
        sensor.AddObservation(distToBall <= kickRange ? 1f : 0f); // 14

        // 15-18: Goal Directions
        Vector3 toOppGoal = transform.InverseTransformPoint(opponentGoal.position).normalized;
        sensor.AddObservation(toOppGoal.x);                     // 15
        sensor.AddObservation(toOppGoal.z);                     // 16
        Vector3 toOwnGoal = transform.InverseTransformPoint(ownGoal.position).normalized;
        sensor.AddObservation(toOwnGoal.x);                     // 17
        sensor.AddObservation(toOwnGoal.z);                     // 18

        // 19-24: Opponent State
        if (opponent != null && opponentRb != null && opponent.gameObject.activeInHierarchy)
        {
            Vector3 oppLocal = transform.InverseTransformPoint(opponent.transform.position);
            sensor.AddObservation(oppLocal.x / 20f);            // 19
            sensor.AddObservation(oppLocal.z / 10f);            // 20
            Vector3 oppVelLocal = transform.InverseTransformDirection(opponentRb.linearVelocity);
            sensor.AddObservation(oppVelLocal.x / 10f);         // 21
            sensor.AddObservation(oppVelLocal.z / 10f);         // 22
            Vector3 oppFwdLocal = transform.InverseTransformDirection(opponent.transform.forward);
            sensor.AddObservation(oppFwdLocal.x);               // 23
            sensor.AddObservation(oppFwdLocal.z);               // 24
        }
        else
        {
            sensor.AddObservation(0f); sensor.AddObservation(0f); sensor.AddObservation(0f);
            sensor.AddObservation(0f); sensor.AddObservation(0f); sensor.AddObservation(0f);
        }
    }

    // ═════════════════════════════════════════════════════════════════
    //  ACTIONS (4 Continuous)
    // ═════════════════════════════════════════════════════════════════
    public override void OnActionReceived(ActionBuffers actions)
    {
        // (Step penalty removed so agent isn't terrified to explore)

        // 2. Parse Actions
        float moveZ     = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f); // Forward/Back
        float moveX     = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f); // Left/Right
        float kickInput = Mathf.Clamp(actions.ContinuousActions[2], 0f, 1f);  // Kick Trigger
        float kickPower = Mathf.Clamp01(actions.ContinuousActions[3]);        // Kick Power

        // 3. Move
        currentMoveDir = (transform.forward * moveZ + transform.right * moveX);
        currentMoveDir.y = 0f;
        currentMoveDir = currentMoveDir.normalized;

        float currentDistToBall = Vector3.Distance(transform.position, ballTransform.position);

        // 4. Kick Action & Ball Touch
        if (currentDistToBall <= kickRange)
        {
            // Massive one-time reward for reaching the ball (breaks the standing-still local minimum)
            if (!touchedBallThisEpisode)
            {
                touchedBallThisEpisode = true;
                AddReward(1.0f);
            }

            if (kickInput > 0.5f && kickCooldown <= 0f)
            {
                float appliedForce = kickPower * kickForce;
                ballRb.AddForce(transform.forward * appliedForce, ForceMode.VelocityChange);
                
                // (Removed the hardcoded +0.1f kick reward here because the agent could trap the ball 
                // and spam kick to farm infinite points. Moving the ball forward naturally triggers 
                // the progress reward below, which is the mathematically safe way to reward kicking).
                
                kickCooldown = 0.5f; // Prevent infinite kick-spam
            }
        }

        // 5. Shaping Reward: Approaching Ball (Strict High-Water Mark)
        // We only reward the agent for reaching a NEW closest distance. 
        if (!touchedBallThisEpisode && currentDistToBall < minRecordDistToBall)
        {
            float approachDelta = minRecordDistToBall - currentDistToBall;
            // HUGE multiplier (0.5f instead of 0.05f). Every step toward the ball 
            // is heavily rewarded. This creates a massive gradient pulling the agent to the ball.
            AddReward(approachDelta * 0.5f);
            minRecordDistToBall = currentDistToBall; // update record
        }

        // 6. Shaping Reward: Pushing Ball to Goal (Strict High-Water Mark)
        float currentBallDistToGoal = Vector3.Distance(ballTransform.position, opponentGoal.position);
        if (currentBallDistToGoal < minRecordBallDistToGoal)
        {
            float progressDelta = minRecordBallDistToGoal - currentBallDistToGoal;
            // MASSIVE multiplier (1.0f instead of 0.2f). Pushing the ball is basically free points.
            AddReward(progressDelta * 1.0f);
            minRecordBallDistToGoal = currentBallDistToGoal; // update record
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuous = actionsOut.ContinuousActions;
        Keyboard kb = Keyboard.current;
        if (kb == null) return;

        continuous[0] = kb.wKey.isPressed ? 1f : kb.sKey.isPressed ? -1f : 0f;
        continuous[1] = kb.dKey.isPressed ? 1f : kb.aKey.isPressed ? -1f : 0f;
        continuous[2] = kb.spaceKey.isPressed ? 1f : 0f;
        continuous[3] = 1.0f; // Full power on manual kick
    }

    // ═════════════════════════════════════════════════════════════════
    //  EVENTS (Goal / Out of Bounds)
    // ═════════════════════════════════════════════════════════════════
    public void GoalScored(Team scoringTeam)
    {
        if (scoringTeam == team)
        {
            // MASSIVE reward for scoring!
            AddReward(GOAL_REWARD);
        }
        else
        {
            // ZERO PENALTY for conceding. Agent does not care about defending!
            AddReward(0.0f); 
        }

        SetSyncedGeneration(GetSyncedGeneration() + 1);
        wantsToEndEpisode = true;
    }

    public void OutOfBounds()
    {
        AddReward(OUT_OF_BOUNDS_PEN);
        SetSyncedGeneration(GetSyncedGeneration() + 1);
        wantsToEndEpisode = true;
        if (opponent != null) opponent.QueueEpisodeEnd();
    }

    public void QueueEpisodeEnd()
    {
        wantsToEndEpisode = true;
    }
}
