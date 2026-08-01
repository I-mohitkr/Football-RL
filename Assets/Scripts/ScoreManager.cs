using UnityEngine;
using TMPro;

public class ScoreManager : MonoBehaviour
{
    public static ScoreManager Instance;

    [Header("=== Score UI ===")]
    public TextMeshPro blueScoreText;
    public TextMeshPro redScoreText;

    private int blueScore = 0;
    private int redScore = 0;

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }
        
        UpdateScoreDisplay();
    }

    public void AddScore(FootballAgent.Team team)
    {
        if (team == FootballAgent.Team.Blue)
        {
            blueScore++;
        }
        else if (team == FootballAgent.Team.Red)
        {
            redScore++;
        }
        
        UpdateScoreDisplay();
    }

    private void UpdateScoreDisplay()
    {
        if (blueScoreText != null)
        {
            blueScoreText.text = blueScore.ToString();
        }
        
        if (redScoreText != null)
        {
            redScoreText.text = redScore.ToString();
        }
    }
}
