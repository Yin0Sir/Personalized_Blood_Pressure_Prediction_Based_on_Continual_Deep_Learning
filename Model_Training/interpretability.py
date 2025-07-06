
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from Model_Def import DDCCor

plt.rcParams.update({
    "font.family": "Arial",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})

def load_sample_from_excel(file_path):
    df = pd.read_excel(file_path)
    ecg = df["ECG"].values
    ppg = df["PPG"].values
    sbp = df["SBP"].dropna().values[0]
    signal = np.stack([ppg, ecg], axis=0)
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    label_tensor = torch.tensor([sbp], dtype=torch.float32)
    return signal_tensor.requires_grad_(), label_tensor, signal

def compute_saliency_map(model, input_tensor, target_value):
    model.eval()
    input_tensor.requires_grad_()
    input_tensor.retain_grad()
    output = model(input_tensor)
    loss = F.mse_loss(output, target_value.unsqueeze(0))
    loss.backward()
    saliency = input_tensor.grad.detach().abs().squeeze(0).numpy()
    return saliency

def plot_saliency(signal, saliency, save_path="saliency_map.png"):
    ecg, ppg = signal
    ecg_sal, ppg_sal = saliency
    L = len(ecg)
    t = np.linspace(0, 10, L)
    r_peaks, _ = find_peaks(ecg, distance=100, height=np.max(ecg) * 0.4)
    ppg_peaks, _ = find_peaks(ppg, distance=100, height=np.max(ppg) * 0.4)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, ecg, label="ECG", color='navy')
    axes[0].fill_between(t, 0, ecg_sal / ecg_sal.max() * np.max(ecg), alpha=0.3, color='salmon', label='Saliency')
    axes[0].scatter(t[r_peaks], ecg[r_peaks], color='red', label='R peaks', s=30, zorder=3)
    axes[0].set_title("ECG Saliency with R-peaks")
    axes[0].legend()
    axes[1].plot(t, ppg, label="PPG", color='darkgreen')
    axes[1].fill_between(t, 0, ppg_sal / ppg_sal.max() * np.max(ppg), alpha=0.3, color='salmon', label='Saliency')
    axes[1].scatter(t[ppg_peaks], ppg[ppg_peaks], color='purple', label='Systolic peaks', s=30, zorder=3)
    axes[1].set_title("PPG Saliency with Systolic Peaks")
    axes[1].legend()
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)

feature_maps = []
gradients = []

def forward_hook(module, input, output):
    feature_maps.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def compute_gradcam(model, input_tensor, label_tensor, save_path="grad_cam.png"):
    _ = model.resnet.layer4.register_forward_hook(forward_hook)
    _ = model.resnet.layer4.register_full_backward_hook(backward_hook)
    feature_maps.clear()
    gradients.clear()
    output = model(input_tensor)
    loss = F.mse_loss(output, label_tensor)
    loss.backward()
    fm = feature_maps[0].detach()
    grad = gradients[0].detach()
    weights = grad.mean(dim=2, keepdim=True)
    cam = (weights * fm).sum(dim=1)
    cam_up = F.interpolate(cam.unsqueeze(1), size=input_tensor.shape[-1], mode='linear').squeeze().cpu().numpy()
    cam_up = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min())
    t = np.linspace(0, 10, len(cam_up))
    plt.figure(figsize=(10, 2.5))
    plt.plot(t, cam_up, label="Grad-CAM Activation", color='darkorange', linewidth=1.5)
    plt.fill_between(t, 0, cam_up, alpha=0.4, color='orange')
    plt.title("Grad-CAM over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)

def visualize_attention(model, save_path="attention_weights_separate.png"):
    n_blocks = len(model.cornet.intlv_layers)
    plt.figure(figsize=(10, 2.2 * n_blocks))
    for i, block in enumerate(model.cornet.intlv_layers):
        ctx = block.saved_context.squeeze(0).cpu().numpy()
        plt.subplot(n_blocks, 1, i + 1)
        plt.plot(ctx, label=f"Block {i+1}", linewidth=1, color='steelblue')
        plt.title(f"CorNet Block {i+1} Context Activations")
        plt.xlabel("Feature Index")
        plt.ylabel("Activation")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)

def compute_occlusion_sensitivity(model, input_tensor, label_tensor, window_size=100, stride=20):
    model.eval()
    B, C, L = input_tensor.shape
    original_pred = model(input_tensor).item()
    errors = []
    positions = []
    for start in range(0, L - window_size + 1, stride):
        occluded = input_tensor.clone()
        occluded[:, :, start:start+window_size] = 0
        pred = model(occluded).item()
        diff = abs(pred - original_pred)
        errors.append(diff)
        positions.append(start + window_size // 2)
    return np.array(positions), np.array(errors)

def plot_occlusion(positions, errors, L, save_path="occlusion_sensitivity.png"):
    t = np.linspace(0, 10, L)
    occlusion_curve = np.zeros(L)
    for pos, err in zip(positions, errors):
        if pos < L:
            occlusion_curve[pos] = err
    plt.figure(figsize=(10, 2.5))
    plt.plot(t, occlusion_curve, color='teal', linewidth=1.5, label="Error due to occlusion")
    plt.fill_between(t, 0, occlusion_curve, alpha=0.4, color='teal')
    plt.title("Occlusion Sensitivity Analysis")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)

def plot_saliency_fft(signal, saliency, save_path="saliency_fft.png"):
    ecg, ppg = signal
    ecg_sal, ppg_sal = saliency
    fs = len(ecg) / 10
    freqs = np.fft.rfftfreq(len(ecg), d=1/fs)
    ecg_power = np.abs(np.fft.rfft(ecg_sal))**2
    ppg_power = np.abs(np.fft.rfft(ppg_sal))**2
    ecg_power /= ecg_power.max()
    ppg_power /= ppg_power.max()
    plt.figure(figsize=(10, 3))
    plt.plot(freqs, ecg_power, label='ECG Saliency Spectrum', color='blue')
    plt.plot(freqs, ppg_power, label='PPG Saliency Spectrum', color='green')
    plt.xlim([0, 10])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Power")
    plt.title("Saliency Frequency Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)

def approx_shap_segments(model, input_tensor, label_tensor, n_segments=20):
    model.eval()
    input_tensor = input_tensor.clone()
    original_output = model(input_tensor).item()
    shap_values = []
    _, _, L = input_tensor.shape
    segment_len = L // n_segments
    for i in range(n_segments):
        x_masked = input_tensor.clone()
        start = i * segment_len
        end = (i + 1) * segment_len if i < n_segments - 1 else L
        x_masked[:, :, start:end] = 0
        output = model(x_masked).item()
        shap_values.append(original_output - output)
    return shap_values

def plot_shap_bar(shap_values, save_path="approx_shap_bar.png"):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(shap_values)), shap_values, color='teal')
    plt.xlabel("Time Segment Index")
    plt.ylabel("Δ Prediction (SHAP approx.)")
    plt.title("Approximate SHAP Value by Temporal Segment")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

def run_all(model_path, sample_path):
    model = DDCCor.DDCCR_Net()
    model = torch.load("PTH/121517/trained_model.pth", map_location=torch.device("cpu"))
    model.eval()
    input_tensor, label_tensor, raw_signal = load_sample_from_excel(sample_path)

    # Saliency
    saliency = compute_saliency_map(model, input_tensor.clone(), label_tensor)
    plot_saliency(raw_signal, saliency)
    # plot_saliency_fft(raw_signal, saliency)

    # Grad-CAM
    compute_gradcam(model, input_tensor.clone(), label_tensor)

    # Attention
    _ = model(input_tensor)
    visualize_attention(model)

    # Occlusion
    pos, err = compute_occlusion_sensitivity(model, input_tensor.clone(), label_tensor)
    plot_occlusion(pos, err, raw_signal.shape[1])

    # SHAP
    # shap_values = approx_shap_segments(model, input_tensor.clone(), label_tensor)
    # plot_shap_bar(shap_values)

    print("✅ All interpretability visualizations generated.")

if __name__ == "__main__":
    run_all("PTH/121517/trained_model.pth", "sample_example.xlsx")
