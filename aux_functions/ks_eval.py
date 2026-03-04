import numpy as np
import matplotlib.pyplot as plt
import os

k=20 # number of snapshots to test for short time and long time
modes = 20 # Need modes strictly less than m/2

# SCORING FOR SHORT-TIME AND LONG-TIME FORECASTS
def scoring_ks(truth, prediction, k, modes):
    """Produce short-time and long-time KS forecast scores.

    Expected input orientation is (space, time):
    - axis 0: spatial index (m points)
    - axis 1: time index (n snapshots)

    Args:
        truth (np.ndarray): Ground-truth KS field with shape (m, n).
        prediction (np.ndarray): Predicted KS field with shape (m, n).
        k (int): Number of snapshots used for scoring. Uses first k columns
            for short-time score and last k columns for long-time score.
        modes (int): Number of centered Fourier modes on each side used in
            long-time spectral comparison. Total selected modes = 2*modes+1.

    Returns:
        tuple[float, float]:
            E1: Short-time score in percent (higher is better).
            E2: Long-time spectral score in percent (higher is better).
    """
    if truth.ndim != 2 or prediction.ndim != 2:
        raise ValueError("truth and prediction must be 2D arrays with shape (space, time).")
    if truth.shape != prediction.shape:
        raise ValueError(
            f"truth and prediction must have identical shape, got {truth.shape} and {prediction.shape}."
        )

    [m,n]=truth.shape
    if k <= 0 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n (n={n}), got k={k}.")
    if modes < 0 or (2 * modes + 1) > m:
        raise ValueError(
            f"modes must satisfy 0 <= modes and 2*modes+1 <= m (m={m}), got modes={modes}."
        )

    Est = np.linalg.norm(truth[:,0:k]-prediction[:,0:k],2)/np.linalg.norm(truth[:,0:k],2)

    m2 = 2*modes+1
    Pt = np.empty((m2,0))
    Pp = np.empty((m2,0))

    # LONG TIME:  Compute least-square fit to power spectra
    for j in range(1, k+1):
        P_truth = np.multiply(np.abs(np.fft.fft(truth[:, n-j])), np.abs(np.fft.fft(truth[:, n-j])))
        P_prediction = np.multiply(np.abs(np.fft.fft(prediction[:, n-j])), np.abs(np.fft.fft(prediction[:, n-j])))
        Pt3 = np.fft.fftshift(P_truth)
        Pp3 = np.fft.fftshift(P_prediction)
        Ptnew = Pt3[int(m/2)-modes:int(m/2)+modes+1]
        Ppnew = Pp3[int(m/2)-modes:int(m/2)+modes+1]  # Fixed the variable name
    
        Pt = np.column_stack((Pt, np.log(Ptnew)))
        Pp = np.column_stack((Pp, np.log(Ppnew)))
    
    Elt = np.linalg.norm(Pt-Pp,2)/np.linalg.norm(Pt,2)
    
    E1 = 100*(1-Est)
    E2 = 100*(1-Elt)

    if np.isnan(E1):
        E1 = -np.inf
    if np.isnan(E2):
        E2 = -np.inf
    
    
    return E1, E2


def instantaneous_energy_per_row(u: np.ndarray) -> np.ndarray:
    """Compute scalar instantaneous energy for each row of a 2D array.

    Uses the same form as in the notebook: 0.5 * mean(u**2, axis=1).

    Args:
        u (np.ndarray): 2D array.

    Returns:
        np.ndarray: 1D energy values with length u.shape[0].
    """
    if u.ndim != 2:
        raise ValueError(f"u must be a 2D array, got shape {u.shape}.")
    return 0.5 * np.mean(u**2, axis=1)


def plot_prediction_energy_over_time(
    prediction: np.ndarray,
    plot_path: str | None = None,
    show_plot: bool = True,
) -> tuple[str | None, np.ndarray]:
    """Plot instantaneous energy versus row index (time-like axis).

    Args:
        prediction (np.ndarray): Prediction array.
        plot_path (str | None): Optional path to save PNG figure.
        show_plot (bool): Whether to display the figure.

    Returns:
        tuple[str | None, np.ndarray]: Saved plot path (or None), energy values.
    """
    energy = instantaneous_energy_per_row(prediction)
    row_index = np.arange(prediction.shape[0])

    plt.figure(figsize=(10, 4))
    plt.plot(row_index, energy, linewidth=2)
    plt.xlabel("Row index")
    plt.ylabel("Instantaneous energy")
    plt.title("Prediction Instantaneous Energy Over Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    saved_plot_path = None
    if plot_path is not None:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=200)
        saved_plot_path = plot_path

    if show_plot:
        plt.show()
    else:
        plt.close()

    return saved_plot_path, energy


def save_prediction_energy_csv(
    prediction: np.ndarray,
    csv_path: str,
    plot_path: str | None = None,
    show_plot: bool = True,
) -> tuple[str, str | None]:
    """Save per-row prediction energy to CSV and plot it.

    Column 1: data_id (integer row index)
    Column 2: prediction (instantaneous energy scalar)
    """
    _, energy = plot_prediction_energy_over_time(
        prediction,
        plot_path=plot_path,
        show_plot=show_plot,
    )

    data_id = np.arange(prediction.shape[0], dtype=int)
    table = np.column_stack((data_id, energy))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header="data_id,prediction",
        comments="",
        fmt=["%d", "%.16f"],
    )
    return csv_path, plot_path


if __name__ == "__main__":
    import os

    DATA_FOLDER = "data"
    TEAM_FOLDER = "team_entries/team0"
    TRUTH_FILE = os.path.join(DATA_FOLDER, "ks_truth.npy")
    PREDICTION_FILE = os.path.join(TEAM_FOLDER, "prediction.npy")
    ENERGY_CSV_FILE = os.path.join(TEAM_FOLDER, "prediction.csv")
    ENERGY_PLOT_FILE = os.path.join(TEAM_FOLDER, "prediction.png")

    truth = np.load(TRUTH_FILE)
    prediction = np.load(PREDICTION_FILE)
    E1, E2 = scoring_ks(truth, prediction, k, modes)
    print("KS evaluation complete.")
    print(f"Truth file: {TRUTH_FILE} | shape={truth.shape}")
    print(f"Prediction file: {PREDICTION_FILE} | shape={prediction.shape}")
    print(f"Scoring parameters: k={k}, modes={modes}")
    print(f"Short-time score (E1, %): {E1:.6f}")
    print(f"Long-time spectral score (E2, %): {E2:.6f}")

    csv_path, plot_path = save_prediction_energy_csv(
        prediction,
        ENERGY_CSV_FILE,
        plot_path=ENERGY_PLOT_FILE,
        show_plot=True,
    )
    print(f"Saved prediction instantaneous-energy CSV: {csv_path}")
    if plot_path is not None:
        print(f"Saved prediction instantaneous-energy plot: {plot_path}")