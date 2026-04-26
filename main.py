# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# from config import *
# from data.prompts import build_prompts
# from models.model_utils import load_model, get_activation
# from sae.sae_model import SAE
# from probe.train_probe import train_probe
# from intervention.apply_direction import apply_intervention
# from eval.bias_metrics import get_bias_score

# # ---------------------------
# # Load model
# # ---------------------------
# model = load_model(DEVICE)

# # ---------------------------
# # Dataset
# # ---------------------------
# prompts = build_prompts(n=120)

# # ---------------------------
# # Collect activations + bias
# # ---------------------------
# acts = []
# bias_scores = []

# for p in prompts:
#     act = get_activation(model, p, LAYER)
#     acts.append(act.detach().cpu().numpy()[0])
#     bias_scores.append(get_bias_score(model, p))

# acts = np.array(acts)
# bias_scores = np.array(bias_scores)

# print("Collected activations:", acts.shape)

# # ---------------------------
# # Train SAE
# # ---------------------------
# d_model = acts.shape[1]
# sae = SAE(d_model, SAE_HIDDEN).to(DEVICE)

# optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
# loss_fn = torch.nn.MSELoss()

# acts_tensor = torch.tensor(acts, dtype=torch.float32).to(DEVICE)

# for epoch in range(EPOCHS):
#     f, recon = sae(acts_tensor)
#     loss = loss_fn(recon, acts_tensor)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print(f"SAE Epoch {epoch}: Loss = {loss.item():.4f}")

# # ---------------------------
# # Extract features
# # ---------------------------
# with torch.no_grad():
#     features, _ = sae(acts_tensor)

# features_np = features.cpu().numpy()

# # ---------------------------
# # Train probe
# # ---------------------------
# direction = train_probe(features_np, bias_scores)

# print("Learned direction shape:", direction.shape)

# # ---------------------------
# # Intervention experiments
# # ---------------------------
# results = []

# for alpha in ALPHAS:
#     new_biases = []

#     for i, p in enumerate(prompts):
#         f = features[i]
#         f_new = apply_intervention(f, direction, alpha)

#         act_new = sae.decoder(f_new).unsqueeze(0)

#         def hook_fn(act, hook):
#             act[:, -1, :] = act_new
#             return act

#         logits = model.run_with_hooks(
#             p,
#             fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", hook_fn)]
#         )

#         he = model.to_single_token(" he")
#         she = model.to_single_token(" she")

#         last = logits[0, -1]
#         new_bias = (last[he] - last[she]).item()
#         new_biases.append(new_bias)

#     avg_bias = np.mean(new_biases)
#     results.append(avg_bias)

#     print(f"Alpha {alpha}: Bias {avg_bias:.4f}")

# # ---------------------------
# # Plot
# # ---------------------------
# plt.plot(ALPHAS, results, marker='o')
# plt.xlabel("Alpha")
# plt.ylabel("Bias Score")
# plt.title("Bias vs Intervention Strength")
# plt.grid()
# plt.savefig("bias_plot.png")
# plt.show()








#experiments = 002


# import os
# import json
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime

# from config import *
# from data.prompts import build_prompts
# from models.model_utils import load_model, get_activation
# from sae.sae_model import SAE
# from probe.train_probe import train_probe
# from intervention.intervene import apply_intervention
# from eval.metrics import get_bias_score

# # ---------------------------
# # Create experiment ID
# # ---------------------------
# exp_id = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
# print(f"Running Experiment: {exp_id}")

# os.makedirs("logs", exist_ok=True)
# os.makedirs("plots", exist_ok=True)

# # ---------------------------
# # Load model
# # ---------------------------
# model = load_model(DEVICE)

# # ---------------------------
# # Dataset
# # ---------------------------
# prompts = build_prompts(n=120)
# print("Number of prompts:", len(prompts))

# # ---------------------------
# # Collect activations + bias
# # ---------------------------
# acts = []
# bias_scores = []

# for p in prompts:
#     act = get_activation(model, p, LAYER)
#     acts.append(act.detach().cpu().numpy()[0])
#     bias_scores.append(get_bias_score(model, p))

# acts = np.array(acts)
# bias_scores = np.array(bias_scores)

# print("Collected activations:", acts.shape)
# print("Bias stats → Mean:", np.mean(bias_scores), "Std:", np.std(bias_scores))

# # ---------------------------
# # Train SAE
# # ---------------------------
# d_model = acts.shape[1]
# sae = SAE(d_model, SAE_HIDDEN).to(DEVICE)

# optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
# loss_fn = torch.nn.MSELoss()

# acts_tensor = torch.tensor(acts, dtype=torch.float32).to(DEVICE)

# loss_history = []

# for epoch in range(EPOCHS):
#     f, recon = sae(acts_tensor)
#     loss = loss_fn(recon, acts_tensor)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     loss_history.append(loss.item())
#     print(f"SAE Epoch {epoch}: Loss = {loss.item():.4f}")

# # ---------------------------
# # Extract features
# # ---------------------------
# with torch.no_grad():
#     features, _ = sae(acts_tensor)

# features_np = features.cpu().numpy()

# # ---------------------------
# # Train probe
# # ---------------------------
# direction = train_probe(features_np, bias_scores)

# print("Direction shape:", direction.shape)

# # ---------------------------
# # Intervention experiments
# # ---------------------------
# results = []

# for alpha in ALPHAS:
#     new_biases = []

#     for i, p in enumerate(prompts):
#         f = features[i]
#         f_new = apply_intervention(f, direction, alpha)

#         act_new = sae.decoder(f_new).unsqueeze(0)

#         def hook_fn(act, hook):
#             act[:, -1, :] = act_new
#             return act

#         logits = model.run_with_hooks(
#             p,
#             fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", hook_fn)]
#         )

#         he = model.to_single_token(" he")
#         she = model.to_single_token(" she")

#         last = logits[0, -1]
#         new_bias = (last[he] - last[she]).item()
#         new_biases.append(new_bias)

#     avg_bias = float(np.mean(new_biases))
#     results.append(avg_bias)

#     print(f"Alpha {alpha}: Bias {avg_bias:.4f}")

# # ---------------------------
# # Plot Bias vs Alpha
# # ---------------------------
# plot_path = f"plots/{exp_id}_bias_plot.png"

# plt.figure()
# plt.plot(ALPHAS, results, marker='o')
# plt.xlabel("Alpha")
# plt.ylabel("Bias Score")
# plt.title("Bias vs Intervention Strength")
# plt.grid()
# plt.savefig(plot_path)
# plt.close()

# print(f"Plot saved to {plot_path}")

# # ---------------------------
# # Save Logs
# # ---------------------------
# log_data = {
#     "experiment_id": exp_id,
#     "config": {
#         "model": MODEL_NAME,
#         "layer": LAYER,
#         "device": DEVICE,
#         "dataset_size": len(prompts),
#         "sae_hidden": SAE_HIDDEN,
#         "epochs": EPOCHS,
#         "lr": LR
#     },
#     "bias_stats": {
#         "mean": float(np.mean(bias_scores)),
#         "std": float(np.std(bias_scores))
#     },
#     "sae": {
#         "loss_history": loss_history,
#         "final_loss": loss_history[-1]
#     },
#     "probe": {
#         "direction_dim": len(direction)
#     },
#     "intervention": {
#         "alphas": ALPHAS,
#         "bias_results": results,
#         "delta_bias": float(results[0] - results[-1])
#     },
#     "plot_path": plot_path
# }

# log_path = f"logs/{exp_id}.json"

# with open(log_path, "w") as f:
#     json.dump(log_data, f, indent=4)

# print(f"Log saved to {log_path}")




# experiment 003 adding multiple layers and saving logs in json format


# import os
# import json
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime

# from config import *
# from data.prompts import build_prompts
# from models.model_utils import load_model, get_activation
# from probe.train_probe import train_probe
# from eval.metrics import get_bias_score

# # ---------------------------
# # Setup experiment
# # ---------------------------
# exp_id = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
# print(f"\nRunning Experiment: {exp_id}")

# os.makedirs("logs", exist_ok=True)
# os.makedirs("plots", exist_ok=True)

# # ---------------------------
# # Load model
# # ---------------------------
# model = load_model(DEVICE)

# # ---------------------------
# # Dataset
# # ---------------------------
# prompts = build_prompts(n=120)
# print("Number of prompts:", len(prompts))

# # ---------------------------
# # Function: run per layer
# # ---------------------------
# def run_experiment_for_layer(model, prompts, layer):

#     acts = []
#     bias_scores = []

#     # ---- Collect activations + bias ----
#     for p in prompts:
#         act = get_activation(model, p, layer)
#         acts.append(act.detach().cpu().numpy()[0])
#         bias_scores.append(get_bias_score(model, p))

#     acts = np.array(acts)
#     bias_scores = np.array(bias_scores)

#     print(f"\nLayer {layer}")
#     print("Activations:", acts.shape)
#     print("Bias Mean:", np.mean(bias_scores), "Std:", np.std(bias_scores))

#     # ---- Train probe (RAW ACTIVATIONS) ----
#     direction = train_probe(acts, bias_scores)

#     # ---- Fix sign ambiguity ----
#     pred = acts @ direction
#     corr = np.corrcoef(pred, bias_scores)[0,1]

#     if corr < 0:
#         direction = -direction
#         print("Flipped direction sign")

#     print("Correlation:", corr)

#     # ---- Intervention ----
#     results = []

#     for alpha in ALPHAS:
#         new_biases = []

#         for i, p in enumerate(prompts):

#             act = torch.tensor(acts[i]).to(DEVICE)
#             dir_t = torch.tensor(direction).to(DEVICE)

#             act_new = act - alpha * dir_t

#             def hook_fn(a, hook):
#                 a[:, -1, :] = act_new
#                 return a

#             logits = model.run_with_hooks(
#                 p,
#                 fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]
#             )

#             he = model.to_single_token(" he")
#             she = model.to_single_token(" she")

#             last = logits[0, -1]
#             new_bias = (last[he] - last[she]).item()
#             new_biases.append(new_bias)

#         avg_bias = float(np.mean(new_biases))
#         results.append(avg_bias)

#         print(f"Alpha {alpha}: Bias {avg_bias:.4f}")

#     delta = results[0] - results[-1]

#     return {
#         "layer": layer,
#         "results": results,
#         "delta_bias": float(delta),
#         "correlation": float(corr),
#         "bias_mean": float(np.mean(bias_scores)),
#         "bias_std": float(np.std(bias_scores))
#     }

# # ---------------------------
# # Run Layer Sweep
# # ---------------------------
# layer_results = []

# for layer in LAYERS_TO_TEST:
#     res = run_experiment_for_layer(model, prompts, layer)
#     layer_results.append(res)

# # ---------------------------
# # Plot comparison
# # ---------------------------
# plot_path = f"plots/{exp_id}_layer_comparison.png"

# plt.figure()

# for res in layer_results:
#     plt.plot(ALPHAS, res["results"], marker='o', label=f"Layer {res['layer']}")

# plt.xlabel("Alpha")
# plt.ylabel("Bias")
# plt.title("Layer-wise Bias Control")
# plt.legend()
# plt.grid()
# plt.savefig(plot_path)
# plt.close()

# print(f"\nPlot saved: {plot_path}")

# # ---------------------------
# # Save logs
# # ---------------------------
# log_data = {
#     "experiment_id": exp_id,
#     "config": {
#         "model": MODEL_NAME,
#         "layers": LAYERS_TO_TEST,
#         "dataset_size": len(prompts),
#         "alphas": ALPHAS
#     },
#     "results": layer_results,
#     "plot": plot_path
# }

# log_path = f"logs/{exp_id}.json"

# with open(log_path, "w") as f:
#     json.dump(log_data, f, indent=4)

# print(f"Log saved: {log_path}")



import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from config import *
from data.prompts import build_prompts
from models.model_utils import load_model, get_activation
from eval.metrics import get_bias_score

# ---------------------------
# Setup experiment
# ---------------------------
exp_id = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
print(f"\nRunning Experiment: {exp_id}")

os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ---------------------------
# Load model
# ---------------------------
model = load_model(DEVICE)

# ---------------------------
# Dataset
# ---------------------------
prompts = build_prompts(n=120)
print("Number of prompts:", len(prompts))


# ---------------------------
# Function: run per layer
# ---------------------------
def run_experiment_for_layer(model, prompts, layer):

    acts = []
    bias_scores = []

    # ---- Collect activations + bias ----
    for p in prompts:
        act = get_activation(model, p, layer)
        acts.append(act.detach().cpu().numpy()[0])
        bias_scores.append(get_bias_score(model, p))

    acts = np.array(acts)
    bias_scores = np.array(bias_scores)

    print(f"\nLayer {layer}")
    print("Activations:", acts.shape)
    print("Bias Mean:", np.mean(bias_scores), "Std:", np.std(bias_scores))

    # ---------------------------
    # 🔥 FIX 1: PCA (reduce dimension)
    # ---------------------------
    pca = PCA(n_components=50)
    acts_reduced = pca.fit_transform(acts)

    # ---------------------------
    # 🔥 FIX 2: Ridge Regression (regularized)
    # ---------------------------
    probe = Ridge(alpha=1.0)
    probe.fit(acts_reduced, bias_scores)

    direction_reduced = probe.coef_

    # ---------------------------
    # Project direction back to original space
    # ---------------------------
    direction = pca.components_.T @ direction_reduced

    # ---------------------------
    # Fix sign ambiguity
    # ---------------------------
    pred = acts @ direction
    corr = np.corrcoef(pred, bias_scores)[0, 1]

    if corr < 0:
        direction = -direction
        corr = -corr
        print("Flipped direction sign")

    print("Correlation:", corr)

    # ---------------------------
    # Intervention
    # ---------------------------
    results = []

    for alpha in ALPHAS:
        new_biases = []

        for i, p in enumerate(prompts):

            act = torch.tensor(acts[i]).to(DEVICE)
            dir_t = torch.tensor(direction).to(DEVICE)

            act_new = act - alpha * dir_t

            def hook_fn(a, hook):
                a[:, -1, :] = act_new
                return a

            logits = model.run_with_hooks(
                p,
                fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]
            )

            he = model.to_single_token(" he")
            she = model.to_single_token(" she")

            last = logits[0, -1]
            new_bias = (last[he] - last[she]).item()
            new_biases.append(new_bias)

        avg_bias = float(np.mean(new_biases))
        results.append(avg_bias)

        print(f"Alpha {alpha}: Bias {avg_bias:.4f}")

    delta = results[0] - results[-1]

    return {
        "layer": layer,
        "results": results,
        "delta_bias": float(delta),
        "correlation": float(corr),
        "bias_mean": float(np.mean(bias_scores)),
        "bias_std": float(np.std(bias_scores))
    }


# ---------------------------
# Run Layer Sweep
# ---------------------------
layer_results = []

for layer in LAYERS_TO_TEST:
    res = run_experiment_for_layer(model, prompts, layer)
    layer_results.append(res)

# ---------------------------
# Plot comparison
# ---------------------------
plot_path = f"plots/{exp_id}_layer_comparison.png"

plt.figure()

for res in layer_results:
    plt.plot(ALPHAS, res["results"], marker='o', label=f"Layer {res['layer']}")

plt.xlabel("Alpha")
plt.ylabel("Bias")
plt.title("Layer-wise Bias Control (PCA + Ridge)")
plt.legend()
plt.grid()
plt.savefig(plot_path)
plt.close()

print(f"\nPlot saved: {plot_path}")

# ---------------------------
# Save logs
# ---------------------------
log_data = {
    "experiment_id": exp_id,
    "config": {
        "model": MODEL_NAME,
        "layers": LAYERS_TO_TEST,
        "dataset_size": len(prompts),
        "alphas": ALPHAS,
        "method": "PCA + Ridge"
    },
    "results": layer_results,
    "plot": plot_path
}

log_path = f"logs/{exp_id}.json"

with open(log_path, "w") as f:
    json.dump(log_data, f, indent=4)

print(f"Log saved: {log_path}")