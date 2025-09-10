import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import time
import collections


def fit_sine_wave(y, p):
    Y = np.fft.fft(y)
    DC = np.real(Y[0]) / p

    if p < 2:
        return lambda x: 0.0, 0.0

    freqs = np.arange(1, p // 2 + 1)
    magnitudes = np.abs(Y[1:p // 2 + 1])
    dominant_index = np.argmax(magnitudes) + 1

    # Special case for Nyquist frequency when p is even.
    if p % 2 == 0 and dominant_index == p // 2:
        a_k = np.real(Y[dominant_index]) / p
        b_k = -np.imag(Y[dominant_index]) / p
    else:
        a_k = 2 * np.real(Y[dominant_index]) / p
        b_k = -2 * np.imag(Y[dominant_index]) / p  # negative from FFT convention

    def fit(x):
        return DC + a_k * np.cos(2 * np.pi * dominant_index * x / p) + b_k * np.sin(2 * np.pi * dominant_index * x / p)

    # Compute R^2
    x_vals = np.arange(p)
    y_fit = fit(x_vals)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    return fit, r2

def memorize_fn(y):
    """Return a function that returns the memorized values of y for integer inputs."""
    def f(x):
        # If x is a scalar, assume it's an integer in [0, p-1]
        if np.isscalar(x):
            return y[int(x)]
        # Otherwise assume x is an array of indices
        return np.array([y[int(xi)] for xi in x])
    return f


def fit_sine_wave_multi_freq(y, p, top_k=1):
    """
    Fit a sum of sine/cosine waves to the signal y using the top K frequencies in its DFT.
    """
    Y = np.fft.fft(y)
    DC = np.real(Y[0]) / p

    # Get magnitudes of frequency components (ignoring DC)
    freqs = np.arange(1, p // 2 + 1)
    magnitudes = np.abs(Y[1:p // 2 + 1])

    # Select top_k frequency indices by magnitude
    top_indices = freqs[np.argsort(magnitudes)[-top_k:]]
    top_indices = np.sort(top_indices)  # Sort for consistency

    # Extract cosine/sine coefficients with special handling for Nyquist frequency.
    coeffs = []
    for k in top_indices:
        if p % 2 == 0 and k == p // 2:
            a_k = np.real(Y[k]) / p
            b_k = -np.imag(Y[k]) / p
        else:
            a_k = 2 * np.real(Y[k]) / p
            b_k = -2 * np.imag(Y[k]) / p  # Negative sign because of FFT convention
        coeffs.append((k, a_k, b_k))

    def fit(x):
        x = np.asarray(x)
        result = np.full_like(x, DC, dtype=np.float64)
        for k, a_k, b_k in coeffs:
            omega = 2 * np.pi * k * x / p
            result += a_k * np.cos(omega) + b_k * np.sin(omega)
        return result

    # Compute R^2
    x_vals = np.arange(p)
    y_fit = fit(x_vals)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0


    return fit, r2

def reconstruct_sine_fits(contrib_a, contrib_b, p, top_k=1):
    """
    Reconstruct sine waves for every neuron from their contribution_a and contribution_b.

    For each neuron:
      - If both contrib_a and contrib_b are "active" (> 0.1), attempt to fit top_k sine/cosine 
        frequencies using DFT.
      - Then, check the DFT of each (a or b) signal. If the second largest frequency
        magnitude is >=30% of the largest, we "memorize" that function (set it to return the raw data).
        For usage and final predictions, we treat that R^2 as 1.0.
        *But* for storing in freq_counts, we still want the genuine R^2 from the initial fit.
      - Otherwise, if the neuron is not active, assign the zero function.

    Additionally, for all active neurons (those that satisfy the above condition), compute the 
    dominant frequency from the DFT of y_a (ignoring DC) and count the number of neurons 
    associated with each frequency, plus the average of their genuine R^2.

    Args:
        contrib_a (np.ndarray): shape (p, num_neurons)
        contrib_b (np.ndarray): shape (p, num_neurons)
        p (int): number of input points (domain: 0 to p-1)
        top_k (int): number of DFT components to use in the fit

    Returns:
        fits_a (list): list of fit functions for contribution_a
        fits_b (list): list of fit functions for contribution_b
        avg_r2 (float): average R² across all valid (active) neurons
        active_count (int): number of neurons that were active (i.e. not zero'd out)
        memorized_count (int): number of neurons for which a memorized function was used
        freq_counts (dict): dictionary mapping each dominant frequency (int)
            to a tuple (count, average_r2) among the neurons that had that frequency.
            The R² used here is always the genuine (pre‐memorization) one.
    """
    def memorize_fn(y):
        """Return a function that memorizes y (assumes x is in [0, p-1])."""
        def f(x):
            if np.isscalar(x):
                return y[int(x)]
            return np.array([y[int(xi)] for xi in x])
        return f

    num_neurons = contrib_a.shape[1]
    fits_a = []
    fits_b = []
    r2_values = []
    active_count = 0
    memorized_count = 0
    
    # We'll accumulate (count, sum_of_r2) for each frequency;
    # later we convert that to (count, avg_r2).
    freq_counts = {}  # dom_freq -> (count, sum_r2)

    for neuron in range(num_neurons):
        y_a = contrib_a[:, neuron]
        y_b = contrib_b[:, neuron]

        # Check if the neuron is "active"
        if np.max(np.abs(y_a)) > 0.1 and np.max(np.abs(y_b)) > 0.1:
            active_count += 1

            # --- First, fit using DFT (top_k frequency components) ---
            fit_a, r2_a = fit_sine_wave_multi_freq(y_a, p, top_k=top_k)
            fit_b, r2_b = fit_sine_wave_multi_freq(y_b, p, top_k=top_k)

            combined_r2 = (r2_a + r2_b) / 2.0  # used for final averaging
            # We also keep the "genuine" combined R² *before* we might overwrite it
            genuine_combined_r2 = combined_r2

            print(f"Neuron {neuron}: R²(a)={r2_a:.6f}, R²(b)={r2_b:.6f}")

            # Check DFT for y_a to find dominant frequency (ignoring DC)
            dft_a = np.fft.fft(y_a)
            mag_a = np.abs(dft_a[1:(p // 2) + 1])  # Exclude DC
            if mag_a.size > 0:
                dom_freq = int(np.argmax(mag_a)) + 1  # +1 to adjust index since we skipped DC
                
                # Update freq_counts with the genuine R²
                count_so_far, sum_r2_so_far = freq_counts.get(dom_freq, (0, 0.0))
                freq_counts[dom_freq] = (count_so_far + 1, sum_r2_so_far + genuine_combined_r2)

            # --- Now check if we should "memorize" (a) or (b) ---
            # If so, we'll override r2_a or r2_b for final usage, but freq_counts
            # keeps the original "genuine" values.
            
            # Check DFT for contrib_a:
            if mag_a.size >= 2:
                largest_a = np.max(mag_a)
                second_largest_a = np.partition(mag_a, -2)[-2]
                if second_largest_a >= 0.30 * largest_a:
                    print(f"Neuron {neuron} (contrib_a): second largest freq magnitude "
                          f"{second_largest_a:.6f} >= 30% of largest {largest_a:.6f}")
                    print("Pay attention here! R²(a):", f"{r2_a:.6f}")
                    print("→ Memorizing contrib_a values instead.")
                    fit_a = memorize_fn(y_a)
                    r2_a = 1.0  # For final usage, we treat it as perfect
                    memorized_count += 1

            # Check DFT for contrib_b:
            dft_b = np.fft.fft(y_b)
            mag_b = np.abs(dft_b[1:(p // 2) + 1])
            if mag_b.size >= 2:
                largest_b = np.max(mag_b)
                second_largest_b = np.partition(mag_b, -2)[-2]
                if second_largest_b >= 0.30 * largest_b:
                    print(f"Neuron {neuron} (contrib_b): second largest freq magnitude "
                          f"{second_largest_b:.6f} >= 30% of largest {largest_b:.6f}")
                    print("Pay attention here! R²(b):", f"{r2_b:.6f}")
                    print("→ Memorizing contrib_b values instead.")
                    fit_b = memorize_fn(y_b)
                    r2_b = 1.0  # For final usage
                    memorized_count += 1

            # The final combined R² for "usage" might have changed from memorization, but
            # for freq_counts, we used the original genuine_combined_r2 above
            # so freq_counts won't be artificially inflated by memorization.
            combined_r2 = (r2_a + r2_b) / 2.0
            r2_values.append(combined_r2)

        else:
            # Inactive neuron, use zero function.
            fit_a = lambda x: np.zeros_like(x, dtype=float) if isinstance(x, np.ndarray) else 0.0
            fit_b = lambda x: np.zeros_like(x, dtype=float) if isinstance(x, np.ndarray) else 0.0

        fits_a.append(fit_a)
        fits_b.append(fit_b)

    # Convert r2_values into an average for all active neurons
    avg_r2 = np.mean(r2_values) if r2_values else 0.0

    # Now convert freq_counts from (count, sum_r2) -> (count, average_r2).
    for freq, (count, sum_r2) in freq_counts.items():
        freq_counts[freq] = (count, sum_r2 / count)

    print("\n======================")
    print(f"Final Average R² value (excluding zero fits): {avg_r2:.6f}")
    print(f"Active neurons: {active_count} out of {num_neurons}")
    print(f"Memorized neurons: {memorized_count}")
    print(f"Frequency distribution (dominant frequencies): {freq_counts}")
    print("======================\n")

    return fits_a, fits_b, avg_r2, active_count, memorized_count, freq_counts

def reconstruct_sine_fits_multilayer(
    contrib_a,
    contrib_b,
    bias_layer1,
    additional_layers_params,  # List (or tuple) of dictionaries for layers 2, 3, etc.
    p,
    top_k=1
):
    """
    Reconstruct the network's behavior for multilayer MLPs.
    
    Returns:
      ...
      dominant_freq_clusters (list): One dict per layer. Each maps frequency -> list of neuron indices.
    """
    import numpy as np

    zero_fn = lambda x: 0.0

    # === Layer 1 Reconstruction ===
    num_neurons_layer1 = contrib_a.shape[1]
    h1 = np.zeros((p, p, num_neurons_layer1))
    layer1_freq = {}
    layer1_fits = []
    dead_neurons_layer1 = []
    layer1_clusters = {}

    for n in range(num_neurons_layer1):
        y_a = contrib_a[:, n]
        y_b = contrib_b[:, n]
        bias_val = bias_layer1[n]

        if np.max(np.abs(y_a)) > 0.1 and np.max(np.abs(y_b)) > 0.1:
            fit_a, r2_a = fit_sine_wave_multi_freq(y_a, p, top_k=top_k)
            fit_b, r2_b = fit_sine_wave_multi_freq(y_b, p, top_k=top_k)
            combined_r2 = (r2_a + r2_b) / 2.0

            Y = np.fft.fft(y_a)
            mag_a = np.abs(Y[1:(p // 2) + 1])
            if mag_a.size > 0:
                dom_freq = int(np.argmax(mag_a)) + 1
                layer1_freq.setdefault(dom_freq, [0, 0.0])
                layer1_freq[dom_freq][0] += 1
                layer1_freq[dom_freq][1] += combined_r2

                # Track per-neuron dominant frequency
                layer1_clusters.setdefault(dom_freq, []).append(n)

            for a in range(p):
                for b in range(p):
                    h1[a, b, n] = np.maximum(fit_a(a) + fit_b(b) + bias_val, 0.0)
            layer1_fits.append((fit_a, fit_b, bias_val))
        else:
            dead_neurons_layer1.append(n)
            for a in range(p):
                for b in range(p):
                    h1[a, b, n] = 0.0
            layer1_fits.append((zero_fn, zero_fn, 0))

    for freq in layer1_freq:
        count, total_r2 = layer1_freq[freq]
        layer1_freq[freq] = [count, total_r2 / count if count > 0 else 0.0]

    dominant_freq_clusters = [layer1_clusters]

    # === Additional Layers Reconstruction ===
    additional_layers_freq = []
    additional_layers_fits_lookup = []
    additional_layers_dead_neurons = []

    current_input = np.maximum(h1, 0)

    for layer_params in additional_layers_params:
        h_pre = np.einsum('abn,nm->abm', current_input, layer_params["kernel"]) + layer_params["bias"]

        layer_freq = {}
        layer_fits_lookup = []
        dead_neurons = []
        layer_clusters = {}

        num_neurons = h_pre.shape[-1]
        for m in range(num_neurons):
            lookup = []
            dom_freq = None

            for a in range(p):
                vector = h_pre[a, :, m]
                if np.max(vector) <= 0.05:
                    lookup.append(zero_fn)
                else:
                    fit_fn, r2 = fit_sine_wave_multi_freq(vector, p, top_k=top_k)
                    lookup.append(fit_fn)

                    Y = np.fft.fft(vector)
                    magnitudes = np.abs(Y[1:(p // 2) + 1])
                    if magnitudes.size > 0:
                        current_freq = int(np.argmax(magnitudes)) + 1
                        dom_freq = current_freq

                        layer_freq.setdefault(current_freq, [0, 0.0, np.zeros(p // 2, dtype=int)])
                        layer_freq[current_freq][0] += 1
                        layer_freq[current_freq][1] += r2

                        top_indices = np.argsort(magnitudes)[-5:]
                        for idx in top_indices:
                            layer_freq[current_freq][2][idx] += 1

            if all(fn is zero_fn for fn in lookup):
                dead_neurons.append(m)

            if dom_freq is not None:
                layer_clusters.setdefault(dom_freq, []).append(m)

            layer_fits_lookup.append(lookup)

        for freq in layer_freq:
            count, total_r2, top5_array = layer_freq[freq]
            layer_freq[freq] = [count, total_r2 / count if count > 0 else 0.0, top5_array.tolist()]

        additional_layers_freq.append(layer_freq)
        additional_layers_fits_lookup.append(layer_fits_lookup)
        additional_layers_dead_neurons.append(dead_neurons)
        dominant_freq_clusters.append(layer_clusters)

        current_input = np.maximum(h_pre, 0)

    return (
        layer1_freq,
        additional_layers_freq,
        layer1_fits,
        additional_layers_fits_lookup,
        dead_neurons_layer1,
        additional_layers_dead_neurons,
        dominant_freq_clusters
    )


def reconstruct_sine_fits_multilayer_logn_fits_layers_after_2(
    contrib_a,
    contrib_b,
    bias_layer1,
    additional_layers_params,  # List (or tuple) of dictionaries for layers 2, 3, etc.
    p,
    top_k=1
):
    """
    Reconstruct the network's behavior for multilayer MLPs.
    Now uses the number of unique layer-1 frequencies as top_k for all deeper layers.

    Returns:
      layer1_freq (dict): freq -> [count, avg_r2]
      additional_layers_freq (list of dict): per-layer freq stats [count, avg_r2, top_indices_counts]
      layer1_fits (list): per-neuron (fit_a, fit_b, bias)
      additional_layers_fits_lookup (list of lists): per-layer, per-neuron list of fit functions over b
      dead_neurons_layer1 (list): indices of dead neurons in layer 1
      additional_layers_dead_neurons (list of lists): per-layer dead neuron indices
      dominant_freq_clusters (list of dict): per-layer freq -> [neuron indices]
    """
    zero_fn = lambda x: 0.0

    # === Layer 1 Reconstruction ===
    num_neurons_layer1 = contrib_a.shape[1]
    h1 = np.zeros((p, p, num_neurons_layer1))
    layer1_freq = {}
    layer1_fits = []
    dead_neurons_layer1 = []
    layer1_clusters = {}

    for n in range(num_neurons_layer1):
        y_a = contrib_a[:, n]
        y_b = contrib_b[:, n]
        bias_val = bias_layer1[n]

        if np.max(np.abs(y_a)) > 0.1 and np.max(np.abs(y_b)) > 0.1:
            fit_a, r2_a = fit_sine_wave_multi_freq(y_a, p, top_k=top_k)
            fit_b, r2_b = fit_sine_wave_multi_freq(y_b, p, top_k=top_k)
            combined_r2 = (r2_a + r2_b) / 2.0

            Y = np.fft.fft(y_a)
            mag_a = np.abs(Y[1:(p // 2) + 1])
            if mag_a.size > 0:
                dom_freq = int(np.argmax(mag_a)) + 1
                layer1_freq.setdefault(dom_freq, [0, 0.0])
                layer1_freq[dom_freq][0] += 1
                layer1_freq[dom_freq][1] += combined_r2
                layer1_clusters.setdefault(dom_freq, []).append(n)

            for a in range(p):
                for b in range(p):
                    h1[a, b, n] = np.maximum(fit_a(a) + fit_b(b) + bias_val, 0.0)

            layer1_fits.append((fit_a, fit_b, bias_val))
        else:
            dead_neurons_layer1.append(n)
            # keeps h1 zeros
            layer1_fits.append((zero_fn, zero_fn, 0.0))

    # finalize layer1_freq averages
    for freq, (count, total_r2) in layer1_freq.items():
        layer1_freq[freq] = [count, total_r2 / count if count > 0 else 0.0]

    # build cluster list and determine new top_k
    dominant_freq_clusters = [layer1_clusters]
    new_top_k = len(layer1_freq)

    # === Additional Layers Reconstruction ===
    additional_layers_freq = []
    additional_layers_fits_lookup = []
    additional_layers_dead_neurons = []
    current_input = np.maximum(h1, 0.0)

    for layer_params in additional_layers_params:
        # pre-activation
        h_pre = np.einsum('abn,nm->abm', current_input, layer_params["kernel"]) + layer_params["bias"]

        layer_freq = {}
        layer_fits_lookup = []
        dead_neurons = []
        layer_clusters = {}
        num_neurons = h_pre.shape[-1]

        for m in range(num_neurons):
            lookup = []
            dom_freq = None

            for a in range(p):
                vector = h_pre[a, :, m]
                if np.max(vector) <= 0.05:
                    lookup.append(zero_fn)
                else:
                    # fit using updated top_k
                    fit_fn, r2 = fit_sine_wave_multi_freq(vector, p, top_k=new_top_k)
                    lookup.append(fit_fn)

                    Y = np.fft.fft(vector)
                    mags = np.abs(Y[1:(p // 2) + 1])
                    if mags.size > 0:
                        current_freq = int(np.argmax(mags)) + 1
                        dom_freq = current_freq

                        # track freq stats
                        layer_freq.setdefault(current_freq, [0, 0.0, np.zeros(p // 2, dtype=int)])
                        layer_freq[current_freq][0] += 1
                        layer_freq[current_freq][1] += r2

                        # count top indices occurrences
                        top_indices = np.argsort(mags)[-new_top_k:]
                        for idx in top_indices:
                            layer_freq[current_freq][2][idx] += 1

            if all(fn is zero_fn for fn in lookup):
                dead_neurons.append(m)
            if dom_freq is not None:
                layer_clusters.setdefault(dom_freq, []).append(m)

            layer_fits_lookup.append(lookup)

        # finalize this layer's freq stats
        for freq, (count, total_r2, idx_counts) in layer_freq.items():
            layer_freq[freq] = [
                count,
                total_r2 / count if count > 0 else 0.0,
                idx_counts.tolist()
            ]

        additional_layers_freq.append(layer_freq)
        additional_layers_fits_lookup.append(layer_fits_lookup)
        additional_layers_dead_neurons.append(dead_neurons)
        dominant_freq_clusters.append(layer_clusters)

        # prepare for next layer
        current_input = np.maximum(h_pre, 0.0)

    return (
        layer1_freq,
        additional_layers_freq,
        layer1_fits,
        additional_layers_fits_lookup,
        dead_neurons_layer1,
        additional_layers_dead_neurons,
        dominant_freq_clusters
    )



def mod_inverse(a, m):
    """
    Compute the modular inverse of a modulo m using the Extended Euclidean Algorithm.
    Returns the inverse if it exists, otherwise raises a ValueError.
    """
    if a == 0:
        raise ValueError("a = 0, Modular inverse does not exist")
    original_m = m
    x0, x1 = 1, 0
    while m:
        q, a, m = a // m, m, a % m
        x0, x1 = x1, x0 - q * x1
    if a != 1:
        raise ValueError("Modular inverse does not exist")
    return x0 % original_m

def plot_cluster_preactivations(cluster_groupings, neuron_data, mlp_class, seed, features, num_neurons, base_dir="plots"):
    import os
    import numpy as np
    import plotly.graph_objs as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    for layer_idx, cluster_group in enumerate(cluster_groupings, start=1):
        output_dir = os.path.join(
            base_dir,
            f"features={features}_num_neurons={num_neurons}",
            "preactivations",
            mlp_class,
            f"layer_{layer_idx}"
        )
        os.makedirs(output_dir, exist_ok=True)

        for freq, neuron_indices in cluster_group.items():
            html_parts = []
            header = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Layer {layer_idx} | Cluster freq = {freq} preactivations for MLP_class = {mlp_class} (seed={seed})</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
"""
            html_parts.append(header)
            description = f"""<p>
This report displays several plots per neuron in layer {layer_idx} with dominant frequency {freq}. 
<ul>
  <li>Plot 1: Real (network) preactivation heatmap vs fitted version.</li>
  <li>Plot 2: DFT of row (a) with max real activation.</li>
  <li>Plot 3: Remapped real preactivation vs remapped fitted.</li>
  <li>Plot 4: Raw and log-scaled 2D FFT magnitude (of real preactivation).</li>
</ul>
</p>
"""
            html_parts.append(description)

            for neuron_idx in neuron_indices:
                if layer_idx not in neuron_data or neuron_idx not in neuron_data[layer_idx]:
                    continue

                data = neuron_data[layer_idx][neuron_idx]
                a_vals = data['a_values']
                b_vals = data['b_values']

                real_preactivation = data.get('real_preactivations', None)
                fitted_preactivation = data.get('fitted_preactivations', None)
                if real_preactivation is None or fitted_preactivation is None:
                    continue

                p_val = len(a_vals)

                # --- Plot 1: Real vs Fitted ---
                max_idx = np.unravel_index(np.argmax(real_preactivation), real_preactivation.shape)
                fixed_a = max_idx[0]
                row_preactivation = real_preactivation[fixed_a, :]
                dft_row = np.fft.fft(row_preactivation)
                freqs = np.arange(p_val // 2 + 1)
                mag_dft = np.abs(dft_row[: len(freqs)])

                fig1 = make_subplots(rows=1, cols=2, subplot_titles=["Real Preactivations", "Fitted from Sines"], horizontal_spacing=0.1)
                fig1.add_trace(go.Heatmap(
                    x=b_vals, y=a_vals, z=real_preactivation,
                    colorscale='viridis',
                    colorbar=dict(title="Activation")
                ), row=1, col=1)
                fig1.add_trace(go.Heatmap(
                    x=b_vals, y=a_vals, z=fitted_preactivation,
                    colorscale='viridis',
                    showscale=False
                ), row=1, col=2)
                fig1.update_layout(
                    title=f"Neuron {neuron_idx} | Real vs Fitted Preactivation",
                    height=400,
                    yaxis=dict(scaleanchor="x"),
                    yaxis2=dict(scaleanchor="x2")
                )

                # --- Plot 2: DFT of row
                fig2 = go.Figure(data=go.Scatter(x=freqs, y=mag_dft, mode="lines+markers"))
                fig2.update_layout(
                    title=f"Neuron {neuron_idx} | DFT of row a={fixed_a}",
                    xaxis_title="Frequency",
                    yaxis_title="Magnitude"
                )

                # --- Plot 3: Remapped
                remapped = np.zeros_like(real_preactivation)
                remapped_fitted = np.zeros_like(fitted_preactivation)
                for a in range(p_val):
                    for b in range(p_val):
                        new_a = (freq * a) % p_val
                        new_b = (freq * b) % p_val
                        remapped[new_a, new_b] = real_preactivation[a, b]
                        remapped_fitted[new_a, new_b] = fitted_preactivation[a, b]

                fig3 = make_subplots(rows=1, cols=2, subplot_titles=["Remapped Real", "Remapped Fitted"], horizontal_spacing=0.1)
                fig3.add_trace(go.Heatmap(
                    z=remapped,
                    colorscale='viridis',
                    colorbar=dict(title="Activation")
                ), row=1, col=1)
                fig3.add_trace(go.Heatmap(
                    z=remapped_fitted,
                    colorscale='viridis',
                    showscale=False
                ), row=1, col=2)
                fig3.update_layout(
                    title=f"Neuron {neuron_idx} | Remapped Real vs Fitted",
                    height=400,
                    yaxis=dict(scaleanchor="x"),
                    yaxis2=dict(scaleanchor="x2")
                )

                # --- Plot 4: FFT
                fft2 = np.fft.fft2(real_preactivation)
                fft_mag = np.abs(fft2)
                fft_log = np.log1p(fft_mag)
                ticks = np.arange(p_val // 2 + 1)
                fft_crop = fft_mag[: p_val // 2 + 1, : p_val // 2 + 1]
                fft_log_crop = fft_log[: p_val // 2 + 1, : p_val // 2 + 1]
                fig4 = make_subplots(rows=1, cols=2, subplot_titles=["|FFT2| (Real)", "log(1 + |FFT2|) (Real)"])
                fig4.add_trace(go.Heatmap(
                    z=fft_crop, x=ticks, y=ticks,
                    colorscale='plasma',
                    colorbar=dict(title="|FFT2|")
                ), row=1, col=1)
                fig4.add_trace(go.Heatmap(
                    z=fft_log_crop, x=ticks, y=ticks,
                    colorscale='plasma',
                    showscale=False
                ), row=1, col=2)
                fig4.update_layout(
                    title=f"Neuron {neuron_idx} | 2D FFT of Real Preactivation (freqs 0..{p_val//2})",
                    height=400,
                    yaxis=dict(scaleanchor="x"),
                    yaxis2=dict(scaleanchor="x2")
                )

                # Append all plots
                for fig in [fig1, fig2, fig3, fig4]:
                    html_parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))
                    html_parts.append("<br><hr><br>\n")

            html_parts.append("</body>\n</html>")
            output_path = os.path.join(output_dir, f"cluster-freq={freq}_preactivations_seed={seed}.html")
            with open(output_path, "w") as f:
                f.write("".join(html_parts))
            print(f"Saved layer {layer_idx} plots for freq {freq} (seed={seed}) to {output_path}")





# --------------------------------------------------------------------------- #
#  Helpers used by both summed_* functions
# --------------------------------------------------------------------------- #
def _iter_layers(cluster_groupings):
    """
    Normalise `cluster_groupings` so that we can iterate uniformly.

    * If the caller passed a single dict  -> treat it as layer 1 only.
    * Otherwise we assume it's already a list/tuple of dicts.
    """
    if isinstance(cluster_groupings, collections.abc.Mapping):
        # single-layer case → wrap inside a list
        return [(1, cluster_groupings)]
    else:
        return [(idx + 1, grp) for idx, grp in enumerate(cluster_groupings)]


# ═══════════════════════════════════════════════════════════════════════════ #
# 1)  summed_preactivations  (multi-layer)
# ═══════════════════════════════════════════════════════════════════════════ #
def summed_preactivations(cluster_groupings,
                          neuron_data,
                          biases_by_layer,
                          mlp_class, seed, features, num_neurons,
                          base_dir="plots"):
    """
    Drop‑in replacement: iterates over each hidden layer, each freq‑cluster,
    sums full preactivations and produces three heatmaps per cluster:
      1) original sum,
      2) remapped by freq,
      3) remapped by modular inverse.
    Saves per‑layer HTML: summed_preactivations_seed={seed}_layer_{ℓ}.html
    """
    for layer_idx, grouping in _iter_layers(cluster_groupings):
        # prepare output directory
        out_dir = os.path.join(
            base_dir,
            f"features={features}_num_neurons={num_neurons}",
            "summed_preactivations",
            mlp_class,
            f"layer_{layer_idx}"
        )
        os.makedirs(out_dir, exist_ok=True)

        bias_vec = biases_by_layer[layer_idx-1]

        # start HTML
        parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Summed Preactivations | layer {layer_idx} (seed={seed})</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  .side-by-side-triple {{ display:inline-block; width:32%; vertical-align:top; margin-right:1%; }}
  .cluster-heading     {{ font-size:1.8em; margin-top:30px; }}
  .section-heading     {{ font-size:2em; margin-top:50px; }}
</style></head><body>
<h1 class="section-heading">Preactivation Summations — Layer {layer_idx}</h1>
"""]

        # for each freq-cluster
        for freq, neuron_idxs in grouping.items():
            parts.append(f'<h2 class="cluster-heading">Cluster freq = {freq}</h2>')
            cluster_sum = None

            # sum full preactivations
            for n in neuron_idxs:
                data = neuron_data[layer_idx][n]
                # layer1: use contrib_a/b + bias, deeper layers: real_preactivations
                if "contrib_a" in data and "contrib_b" in data:
                    full = np.add.outer(data["contrib_a"], data["contrib_b"]) + bias_vec[n]
                else:
                    full = data["real_preactivations"]

                cluster_sum = full if cluster_sum is None else cluster_sum + full

            p = cluster_sum.shape[0]
            a_vals = b_vals = np.arange(p)
            mag = np.abs(cluster_sum)

            # helper to remap a matrix
            def remap(mat, factor):
                out = np.empty_like(mat)
                for i in range(p):
                    for j in range(p):
                        out[(factor*i) % p, (factor*j) % p] = mat[i, j]
                return out

            # original
            fig0 = go.Figure(go.Heatmap(x=b_vals, y=a_vals, z=mag, colorscale='viridis'))
            fig0.update_layout(title="Original Summed Preactivation")
            html0 = pio.to_html(fig0, full_html=False, include_plotlyjs=False)

            # remapped by freq
            fig1 = go.Figure(go.Heatmap(z=remap(mag, freq), colorscale='viridis'))
            fig1.update_layout(title=f"Remapped × {freq}")
            html1 = pio.to_html(fig1, full_html=False, include_plotlyjs=False)

            # remapped by modular inverse
            try:
                inv = mod_inverse(freq, p)
                fig2 = go.Figure(go.Heatmap(z=remap(mag, inv), colorscale='viridis'))
                fig2.update_layout(title=f"Remapped × inv({freq})={inv}")
                html2 = pio.to_html(fig2, full_html=False, include_plotlyjs=False)
            except ValueError:
                html2 = "<p>Modular inverse not available.</p>"

            parts += [
                '<div class="side-by-side-triple">' + html0 + '</div>',
                '<div class="side-by-side-triple">' + html1 + '</div>',
                '<div class="side-by-side-triple">' + html2 + '</div>',
                '<br style="clear:both;"><hr><br>'
            ]

        parts.append("</body></html>")

        # write file
        fname = f"summed_preactivations_seed={seed}_layer_{layer_idx}.html"
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write("".join(parts))
        print(f"Saved summed_preactivations → {fname}")


# ──────────────────────────────────────────────────────────────────────────────
def summed_postactivations(cluster_groupings,
                           neuron_data,
                           biases_by_layer,
                           mlp_class, seed, features, num_neurons,
                           base_dir="plots"):
    """
    Same as summed_preactivations, but applies ReLU before summing.
    Saves per‑layer HTML: summed_postactivations_seed={seed}_layer_{ℓ}.html
    Also writes a JSON file with remapped plot data ("Remapped × f").
    """
    import os
    import json
    import numpy as np
    import plotly.graph_objs as go
    import plotly.io as pio

    for layer_idx, grouping in _iter_layers(cluster_groupings):
        out_dir = os.path.join(
            base_dir,
            f"features={features}_num_neurons={num_neurons}",
            "summed_postactivations",
            mlp_class,
            f"layer_{layer_idx}"
        )
        os.makedirs(out_dir, exist_ok=True)

        bias_vec = biases_by_layer[layer_idx-1]

        parts = [f"""<!DOCTYPE html>
<html><head><meta charset=\"utf-8\">   
<title>Summed Postactivations | layer {layer_idx} (seed={seed})</title>
<script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
<style>
  .side-by-side-triple {{ display:inline-block; width:32%; vertical-align:top; margin-right:1%; }}
  .cluster-heading     {{ font-size:1.8em; margin-top:30px; }}
</style></head><body>
<h1>Postactivation Summations — Layer {layer_idx}</h1>
"""]

        # prepare JSON container for remapped data
        remapped_json = {
            "seed": seed,
            "layer": layer_idx,
            "clusters": {}
        }

        for freq, neuron_idxs in grouping.items():
            parts.append(f'<h2 class="cluster-heading">Cluster freq = {freq}</h2>')
            cluster_sum = None

            for n in neuron_idxs:
                data = neuron_data[layer_idx][n]
                if "contrib_a" in data and "contrib_b" in data:
                    pre = np.add.outer(data["contrib_a"], data["contrib_b"]) + bias_vec[n]
                else:
                    pre = data["real_preactivations"]
                post = np.maximum(pre, 0.0)

                cluster_sum = post if cluster_sum is None else cluster_sum + post

            p = cluster_sum.shape[0]
            a_vals = b_vals = np.arange(p)
            mag = np.abs(cluster_sum)

            def remap(mat, factor):
                out = np.empty_like(mat)
                for i in range(p):
                    for j in range(p):
                        out[(factor*i) % p, (factor*j) % p] = mat[i, j]
                return out

            # Original summed postactivation
            fig0 = go.Figure(go.Heatmap(x=b_vals, y=a_vals, z=mag, colorscale='viridis'))
            fig0.update_layout(title="Original Summed Postactivation")
            html0 = pio.to_html(fig0, full_html=False, include_plotlyjs=False)

            # Remapped × freq
            remapped = remap(mag, freq)
            fig1 = go.Figure(go.Heatmap(z=remapped, colorscale='viridis'))
            title1 = f"Remapped × {freq}"
            fig1.update_layout(title=title1)
            html1 = pio.to_html(fig1, full_html=False, include_plotlyjs=False)

            # Attempt remap × inv(freq)
            try:
                inv = mod_inverse(freq, p)
                remapped_inv = remap(mag, inv)
                fig2 = go.Figure(go.Heatmap(z=remapped_inv, colorscale='viridis'))
                title2 = f"Remapped × inv({freq})={inv}"
                fig2.update_layout(title=title2)
                html2 = pio.to_html(fig2, full_html=False, include_plotlyjs=False)
            except ValueError:
                title2 = "Remapped × inv not available"
                remapped_inv = None
                html2 = "<p>Modular inverse not available.</p>"

            parts += [
                '<div class="side-by-side-triple">' + html0 + '</div>',
                '<div class="side-by-side-triple">' + html1 + '</div>',
                '<div class="side-by-side-triple">' + html2 + '</div>',
                '<br style="clear:both;"><hr><br>'
            ]

            # populate JSON only for the "Remapped × freq" plot
            data_list = []
            for i in range(p):
                for j in range(p):
                    data_list.append({
                        "a": int(i),
                        "b": int(j),
                        "value": float(remapped[i, j])
                    })
            remapped_json["clusters"][str(freq)] = {
                "title": title1,
                "data": data_list
            }

        parts.append("</body></html>")

        # write HTML
        fname = f"summed_postactivations_seed={seed}_layer_{layer_idx}.html"
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write("".join(parts))
        print(f"Saved summed_postactivations → {fname}")

        # write JSON
        json_fname = fname.replace('.html', '.json')
        with open(os.path.join(out_dir, json_fname), 'w') as jf:
            json.dump(remapped_json, jf, indent=2)
        print(f"Saved remapped data → {json_fname}")




def plot_cluster_to_logits(cluster_groupings, neuron_data, biases_last_layer,
                           final_layer_weights,
                           mlp_class: str, seed: int, features: int, num_neurons: int,
                           base_dir: str = "plots"):
    """
    Plot only the *last* layer’s clusters’ contributions to the output logits.

    Parameters:
      cluster_groupings: either
        - a single dict for layer 1, or
        - a list/tuple of dicts (one per layer).
      neuron_data: dict mapping layer_idx -> {neuron_idx -> data dict}
      biases_last_layer: 1D array of biases for the *last* hidden layer
      final_layer_weights: array (num_neurons_last, p)
      mlp_class, seed, features, num_neurons, base_dir: as before
    """
    import os
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    # ── Determine the last layer index and its grouping ─────────────── #
    if isinstance(cluster_groupings, dict):
        last_layer_idx = 1
        last_groups = cluster_groupings
    else:
        last_layer_idx = len(cluster_groupings)
        last_groups = cluster_groupings[-1]

    # ── Build output directory ──────────────────────────────────────── #
    output_dir = os.path.join(
        base_dir,
        f"features={features}_num_neurons={num_neurons}",
        "cluster_to_logits",
        mlp_class,
        f"layer_{last_layer_idx}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # ── Shortcut to this layer’s neuron_data ────────────────────────── #
    layer_data = neuron_data[last_layer_idx]

    # ── Loop over each cluster frequency ────────────────────────────── #
    for freq, neuron_indices in last_groups.items():
        html_parts = []

        # ---- HTML header ----
        header = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Layer {last_layer_idx} Cluster→Logits: freq={freq} (seed={seed})</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    .side-by-side-triple {{ display: inline-block; width:32%; vertical-align:top; margin-right:1%; }}
    .dft-container {{ text-align:center; margin-bottom:20px; }}
  </style>
</head>
<body>
<h1>Layer {last_layer_idx} — Cluster freq = {freq}</h1>
<p>Below: the DFTs for up to three neurons in this cluster, then per-logit heatmaps.</p>
"""
        html_parts.append(header)

        # ==== DFT plots for up to 3 neurons ==== #
        selected = []
        for n in neuron_indices:
            data = layer_data[n]
            # only if this layer_data has contrib_a/b
            if 'contrib_a' in data and 'contrib_b' in data:
                max_pre = max(np.max(np.abs(data['contrib_a'])),
                              np.max(np.abs(data['contrib_b'])))
                if max_pre >= 0.09:
                    selected.append(n)
                if len(selected) == 3:
                    break

        if selected:
            n_plots = 2 * len(selected)
            titles = []
            for n in selected:
                titles += [f"Neuron {n}: contrib_a", f"Neuron {n}: contrib_b"]
            fig_dft = make_subplots(rows=1, cols=n_plots, subplot_titles=titles)
            p_val = len(layer_data[selected[0]]['contrib_a'])
            max_idx = p_val // 2 + 1
            freqs = np.arange(max_idx)

            col = 1
            for n in selected:
                d = layer_data[n]
                # FFT of contrib_a
                mag_a = np.abs(np.fft.fft(d['contrib_a'])[:max_idx])
                fig_dft.add_trace(go.Scatter(x=freqs, y=mag_a, mode='lines'),
                                  row=1, col=col)
                col += 1
                # FFT of contrib_b
                mag_b = np.abs(np.fft.fft(d['contrib_b'])[:max_idx])
                fig_dft.add_trace(go.Scatter(x=freqs, y=mag_b, mode='lines'),
                                  row=1, col=col)
                col += 1

            fig_dft.update_layout(
                height=300,
                width=300 * n_plots,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            html_parts.append(f"<div class='dft-container'>{pio.to_html(fig_dft, full_html=False, include_plotlyjs=False)}</div>\n")
        else:
            html_parts.append("<p>No neurons with max preactivation ≥0.09 for DFT.</p>\n")

        # ==== Prepare per-logit accumulation ==== #
        # Assume all neurons share same a/b grid from the first neuron in this cluster
        sample = layer_data[neuron_indices[0]]
        a_vals = sample['a_values']
        b_vals = sample['b_values']
        p = len(a_vals)

        # Initialize sum per logit
        cluster_logit_contrib = {j: np.zeros((p, p)) for j in range(p)}

        # Sum over neurons
        for n in neuron_indices:
            d = layer_data[n]
            # if we have explicit postactivations, use them
            if 'postactivations' in d:
                post = d['postactivations']
            else:
                # first-layer style: rebuild from contrib_a/b + bias
                pre = np.add.outer(d['contrib_a'], d['contrib_b']) + biases_last_layer[n]
                post = np.maximum(0, pre)
            w = final_layer_weights[n]  # shape (p,)
            for j in range(p):
                cluster_logit_contrib[j] += w[j] * post

        # ==== For each logit: three heatmaps ==== #
        def remap(mat, factor):
            out = np.empty_like(mat)
            for i in range(p):
                for k in range(p):
                    out[(factor * i) % p, (factor * k) % p] = mat[i, k]
            return out

        for j in range(p):
            html_parts.append(f"<h2>Logit {j}</h2>\n")

            # a) Normal
            fig_norm = go.Figure(go.Heatmap(
                x=b_vals, y=a_vals, z=cluster_logit_contrib[j],
                colorscale='viridis', colorbar=dict(title="Magnitude")
            ))
            fig_norm.update_layout(title="Normal", height=400, width=400)
            norm_html = pio.to_html(fig_norm, full_html=False, include_plotlyjs=False)

            # b) Remapped by freq
            rem_f = remap(cluster_logit_contrib[j], freq)
            fig_remap = go.Figure(go.Heatmap(
                x=list(range(p)), y=list(range(p)), z=rem_f,
                colorscale='viridis', colorbar=dict(title="Magnitude")
            ))
            fig_remap.update_layout(title=f"Remapped ×{freq}", height=400, width=400)
            remap_html = pio.to_html(fig_remap, full_html=False, include_plotlyjs=False)

            # c) Remapped by modular inverse
            try:
                inv = mod_inverse(freq, p)
                rem_i = remap(cluster_logit_contrib[j], inv)
                fig_inv = go.Figure(go.Heatmap(
                    x=list(range(p)), y=list(range(p)), z=rem_i,
                    colorscale='viridis', colorbar=dict(title="Magnitude")
                ))
                fig_inv.update_layout(title=f"Remapped ×inv({freq})={inv}", height=400, width=400)
                inv_html = pio.to_html(fig_inv, full_html=False, include_plotlyjs=False)
            except ValueError:
                inv_html = "<p>No modular inverse available.</p>"

            # Side‑by‑side trio
            html_parts.append('<div class="side-by-side-triple">' + norm_html   + '</div>')
            html_parts.append('<div class="side-by-side-triple">' + remap_html  + '</div>')
            html_parts.append('<div class="side-by-side-triple">' + inv_html    + '</div>')
            html_parts.append('<br style="clear:both;"><hr><br>\n')

        # ---- HTML footer & write file ----
        html_parts.append("</body>\n</html>")
        full_html = "".join(html_parts)
        fname = f"cluster-to-logits_freq={freq}_seed={seed}.html"
        with open(os.path.join(output_dir, fname), "w") as f:
            f.write(full_html)
        print(f"Saved: {fname}")



def plot_all_clusters_to_logits(
    neuron_data: dict,
    final_layer_weights: np.ndarray,
    mlp_class: str,
    seed: int,
    features: int,
    num_neurons: int,
    base_dir: str = "plots"
):
    """
    Aggregates contributions from *the last hidden layer* to the final output logits.
    Writes one HTML file with a heatmap for each logit.
    
    Parameters:
      neuron_data: dict[layer_idx] -> dict[neuron_idx] -> {
          'a_values', 'b_values', 'real_preactivations', … }
      final_layer_weights: array of shape (num_neurons_last, p)
      mlp_class, seed, features, num_neurons, base_dir: as before
    """
    import os
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    # 1) pick the last layer
    last_layer_idx = max(neuron_data.keys())
    layer_dict = neuron_data[last_layer_idx]

    # 2) build output directory
    output_dir = os.path.join(
        base_dir,
        f"features={features}_num_neurons={num_neurons}",
        "all_clusters_to_logits",
        mlp_class,
        f"layer_{last_layer_idx}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # 3) get the a/b grid
    sample = next(iter(layer_dict.values()))
    a_vals = sample["a_values"]
    b_vals = sample["b_values"]
    p = len(a_vals)

    # 4) accumulate per-logit contributions
    aggregated_contrib = {j: np.zeros((p, p)) for j in range(p)}
    for neuron_idx, data in layer_dict.items():
        # postactivation = ReLU(real preactivations)
        post = np.maximum(data["real_preactivations"], 0.0)
        weights = final_layer_weights[neuron_idx]  # shape (p,)
        for j in range(p):
            aggregated_contrib[j] += weights[j] * post

    # 5) build the HTML
    html_parts = [f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>All Clusters → Logits (layer {last_layer_idx}) | {mlp_class} (seed={seed})</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
<h1>All Clusters to Logits (Layer {last_layer_idx})</h1>
"""]

    for j in range(p):
        fig = go.Figure(
            go.Heatmap(
                x=b_vals,
                y=a_vals,
                z=aggregated_contrib[j],
                colorscale="viridis",
                colorbar=dict(title="Magnitude")
            )
        )
        fig.update_layout(
            title=f"Logit {j}",
            xaxis_title="b",
            yaxis_title="a",
            height=500,
            width=500
        )
        html_parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))
        html_parts.append("<br><hr><br>\n")

    html_parts.append("</body>\n</html>")
    full_html = "".join(html_parts)

    # 6) write it out
    filename = f"all_clusters_to_logits_seed={seed}_layer_{last_layer_idx}.html"
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(full_html)

    print(f"Saved aggregated logits plots (seed={seed}) to {filename}")
