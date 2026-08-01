using UnityEngine;

/// <summary>
/// GameSettings — Attach to an empty GameObject in your scene.
/// Controls FPS cap and physics settings for optimal performance.
/// </summary>
public class GameSettings : MonoBehaviour
{
    [Header("=== Performance ===")]
    [Tooltip("Target frame rate. Set to 60 for MacBook Air.")]
    public int targetFPS = 60;

    [Tooltip("Enable V-Sync (locks to monitor refresh rate). Overrides targetFPS.")]
    public bool useVSync = true;

    private void Awake()
    {
        // Force V-Sync OFF so targetFrameRate works (VSync overrides it)
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = targetFPS;
    }

    private void Start()
    {
        // Re-apply in Start in case anything overrides it
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = targetFPS;
    }
}
