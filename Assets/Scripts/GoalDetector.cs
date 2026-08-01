using UnityEngine;
using System.Collections;

/// <summary>
/// GoalDetector — Attach this to each goal's trigger zone (an invisible Cube
/// with "Is Trigger" checked on its Box Collider).
///
/// Setup:
///   1. Create a Cube inside each goalpost, scale it to fill the goal opening.
///   2. Remove or disable the Mesh Renderer (make it invisible).
///   3. On the Box Collider, check "Is Trigger" = true.
///   4. Tag the ball as "Ball".
///   5. Assign both agents (blueAgent and redAgent) in the Inspector.
///   6. Set goalBelongsTo = Blue if this is the Blue team's goal (the one Blue defends).
/// </summary>
public class GoalDetector : MonoBehaviour
{
    [Header("=== Goal Setup ===")]
    [Tooltip("Which team DEFENDS this goal? If Blue defends it, a ball entering means Red scored.")]
    public FootballAgent.Team goalBelongsTo = FootballAgent.Team.Blue;

    [Header("=== Agent References ===")]
    [Tooltip("The Blue team agent.")]
    public FootballAgent blueAgent;

    [Tooltip("The Red team agent.")]
    public FootballAgent redAgent;

    private bool isProcessingGoal = false;

    private void OnTriggerEnter(Collider other)
    {
        // Only react to the ball, and only if we aren't already waiting to reset
        if (isProcessingGoal || !other.CompareTag("Ball")) return;

        StartCoroutine(ProcessGoalDelay());
    }

    private IEnumerator ProcessGoalDelay()
    {
        isProcessingGoal = true;

        // Determine which team scored
        FootballAgent.Team scoringTeam;
        if (goalBelongsTo == FootballAgent.Team.Blue)
        {
            scoringTeam = FootballAgent.Team.Red;
            Debug.Log("<color=red>⚽ GOAL! Red team scores!</color>");
        }
        else
        {
            scoringTeam = FootballAgent.Team.Blue;
            Debug.Log("<color=cyan>⚽ GOAL! Blue team scores!</color>");
        }

        // Update the scorecard UI!
        if (ScoreManager.Instance != null)
        {
            ScoreManager.Instance.AddScore(scoringTeam);
        }

        // Wait for 1.5 seconds to let the ball hit the net and look natural
        yield return new WaitForSeconds(1.5f);

        // Notify agents and reset
        blueAgent.GoalScored(scoringTeam);
        redAgent.GoalScored(scoringTeam);

        isProcessingGoal = false;
    }
}
