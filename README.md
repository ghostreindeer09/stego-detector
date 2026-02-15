<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Modular Steganalysis Research Framework</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 2rem; background-color: #f9f9f9; color: #333; }
        h1, h2, h3, h4 { color: #2c3e50; }
        code, pre { background-color: #eee; padding: 2px 5px; border-radius: 4px; }
        pre { padding: 10px; overflow-x: auto; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 1rem; }
        th, td { border: 1px solid #ccc; padding: 0.5rem; text-align: left; }
        th { background-color: #f0f0f0; }
        ul, ol { margin: 0 0 1rem 1.5rem; }
        .badge { display: inline-block; margin-right: 0.5rem; }
        hr { border: 0; border-top: 1px solid #ccc; margin: 2rem 0; }
        .code-block { background: #272822; color: #f8f8f2; padding: 1rem; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>

<h1>Modular Steganalysis Research Framework</h1>

<p>
    <img class="badge" src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="Python 3.8+">
    <img class="badge" src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style: Black">
</p>

<p>
A production-quality, config-driven framework for steganographic image analysis research. Supports automated dataset creation, CNN and classical ML training, comprehensive evaluation, robustness testing, and optional interpretability — all from a single YAML config.
</p>

<hr>

<h2>🏗️ Architecture</h2>

<pre class="code-block">
Pro/
├── stego/                          # Core model modules (PRESERVED — existing SRNet pipeline)
│   ├── model.py                    #   SRNet, SRNetBackbone, GradCAM
│   ├── features.py                 #   HPF, SRM, DCT feature extraction
│   ├── datasets.py                 #   Pair-constraint dataset classes
│   ├── detector.py                 #   StegoDetector (inference + Grad-CAM)
│   └── metrics.py                  #   Weighted AUC, F1, confusion matrix
│
├── framework/                      # NEW — Modular research framework
│   ├── config.py                   #   YAML config loader with deep-merge
│   ├── embedding.py                #   LSB embedder + extensible base class
│   ├── dataset_generator.py        #   Automated stego dataset creation
│   ├── dataset_loader.py           #   CSV-based DataLoader factory
│   ├── feature_extractor.py        #   Classical ML features (histogram, GLCM, DCT, etc.)
│   ├── trainer.py                  #   CNNTrainer + ClassicalMLTrainer
│   ├── evaluator.py                #   Comprehensive metrics + comparison tables
│   ├── robustness.py               #   Perturbation sweep testing
│   ├── interpretability.py         #   Grad-CAM + SHAP analysis
│   ├── plotting.py                 #   Publication-quality plots
│   └── tracking.py                 #   MLflow / W&B experiment tracking
│
├── configs/                        # Experiment configurations
│   ├── default_experiment.yaml     #   Full baseline config
│   └── robustness_sweep.yaml       #   Robustness-focused sweep config
│
├── train.py                        # PRESERVED — Original SRNet training script
├── train_alaska2.py                # PRESERVED — Original ALASKA2 training
├── evaluate_confusion_matrix.py    # PRESERVED — Original evaluation
├── app.py                          # PRESERVED — Streamlit demo app
│
├── generate_dataset.py             # NEW — Dataset generation CLI
├── run_experiment.py               # NEW — Main experiment runner
│
├── Dockerfile                      # NEW — Docker setup
├── docker-compose.yml              # NEW — Docker Compose
├── requirements.txt                # Updated dependencies
└── README.md                       # This file
</pre>

<hr>

<h2>🚀 Quick Start</h2>

<h3>1. Install Dependencies</h3>
<pre class="code-block">pip install -r requirements.txt</pre>

<h3>2. Generate a Stego Dataset</h3>
<p>Place your cover images in <code>cover_images/</code>, then run:</p>
<pre class="code-block">
# Using CLI arguments:
python generate_dataset.py --cover-dir cover_images/ --stego-dir stego_images/

# Or using YAML config:
python generate_dataset.py --config configs/default_experiment.yaml
</pre>
<p>This creates:</p>
<ul>
    <li><code>stego_images/</code> — stego versions of each cover image</li>
    <li><code>dataset_mapping.csv</code> — CSV with <code>image_name, label, payload, payload_length, source_path</code></li>
</ul>

<h3>3. Run an Experiment</h3>
<pre class="code-block">
python run_experiment.py --config configs/default_experiment.yaml
</pre>
<p>This will:</p>
<ol>
    <li>✅ Generate dataset (if CSV doesn't exist)</li>
    <li>✅ Load and split data (train/val/test)</li>
    <li>✅ Train SRNet with AMP, label smoothing, early stopping</li>
    <li>✅ Evaluate on test set (precision, recall, F1, AUC, confusion matrix)</li>
    <li>✅ Save plots and metrics to <code>outputs/</code></li>
</ol>

<h3>4. Skip Steps</h3>
<pre class="code-block">
# Skip dataset generation (use existing CSV):
python run_experiment.py --config configs/default_experiment.yaml --skip-generation

# Skip training (evaluate only with existing checkpoint):
python run_experiment.py --config configs/default_experiment.yaml --skip-training

# Skip robustness testing:
python run_experiment.py --config configs/default_experiment.yaml --skip-robustness
</pre>

<hr>

<h2>🐳 Docker</h2>
<pre class="code-block">
# Build
docker build -t steganalysis .

# Run experiment
docker run -v $(pwd)/cover_images:/app/cover_images \
           -v $(pwd)/outputs:/app/outputs \
           steganalysis python run_experiment.py --config configs/default_experiment.yaml

# Or use Docker Compose
docker-compose up
</pre>

<hr>

<h2>⚙️ Configuration (YAML)</h2>
<p>All experiment parameters live in a single YAML file. See <code>configs/default_experiment.yaml</code> for full reference.</p>

<table>
<thead>
<tr>
<th>Section</th>
<th>Controls</th>
</tr>
</thead>
<tbody>
<tr><td><code>experiment</code></td><td>Name, seed, output directory</td></tr>
<tr><td><code>dataset</code></td><td>Cover/stego dirs, embedding method, payload settings</td></tr>
<tr><td><code>dataloader</code></td><td>Batch size, splits, augmentations</td></tr>
<tr><td><code>model</code></td><td>Architecture (srnet/classical_ml), hyperparameters</td></tr>
<tr><td><code>training</code></td><td>Epochs, optimizer, scheduler, AMP, early stopping</td></tr>
<tr><td><code>evaluation</code></td><td>Threshold, metrics to compute, plots</td></tr>
<tr><td><code>robustness</code></td><td>JPEG/noise/resize/crop/payload perturbation sweeps</td></tr>
<tr><td><code>interpretability</code></td><td>Grad-CAM and SHAP settings</td></tr>
<tr><td><code>tracking</code></td><td>MLflow / W&B integration</td></tr>
</tbody>
</table>

<h3>Override Configs</h3>
<pre class="code-block">
python run_experiment.py --config configs/default_experiment.yaml \
                         --override configs/robustness_sweep.yaml
</pre>

<hr>

<h2>📊 Metrics Tracked</h2>
<table>
<thead>
<tr><th>Metric</th><th>Description</th></tr>
</thead>
<tbody>
<tr><td><b>Precision</b></td><td>TP / (TP + FP) — How many detected stego are correct</td></tr>
<tr><td><b>Recall</b></td><td>TP / (TP + FN) — How many stego images are found</td></tr>
<tr><td><b>F1 Score</b></td><td>Harmonic mean of precision and recall</td></tr>
<tr><td><b>ROC-AUC</b></td><td>Area under ROC curve</td></tr>
<tr><td><b>Accuracy</b></td><td>Overall correct classifications</td></tr>
<tr><td><b>Specificity</b></td><td>TN / (TN + FP) — True negative rate</td></tr>
<tr><td><b>FP/FN Counts</b></td><td>Raw false positive and false negative counts</td></tr>
</tbody>
</table>

<hr>

<h2>🔬 Robustness Testing</h2>
<pre class="code-block">
robustness:
  perturbations:
    jpeg_compression:
      qualities: [50, 70, 80, 90, 95]
    gaussian_noise:
      sigmas: [1.0, 5.0, 10.0]
    resize:
      scales: [0.5, 0.75, 1.5]
    crop:
      ratios: [0.5, 0.7, 0.9]
    payload_size:
      lengths: [4, 16, 64, 128]
</pre>

<hr>

<h2>🧪 Research Experiments (TODOs)</h2>
<ol>
<li><b>Payload size robustness</b> — Re-embed with different payload lengths and measure detection rate</li>
<li><b>Classical ML pipeline</b> — Extract features from DataLoader and train sklearn models</li>
<li><b>Model comparison</b> — Run multiple architectures and use <code>plot_model_comparison()</code></li>
<li><b>SHAP analysis</b> — Install <code>shap</code> and use <code>SHAPAnalyzer</code> for feature importance</li>
<li><b>Custom embedding methods</b> — Subclass <code>BaseEmbedder</code> in <code>framework/embedding.py</code></li>
<li><b>Cross-dataset transfer</b> — Train on ALASKA2, test on BOSSBase</li>
</ol>

<hr>

<h2>📈 Optional: Experiment Tracking</h2>

<h3>MLflow</h3>
<pre class="code-block">pip install mlflow</pre>
<p>Set in config:</p>
<pre class="code-block">
tracking:
  enabled: true
  backend: "mlflow"
</pre>

<h3>Weights & Biases</h3>
<pre class="code-block">pip install wandb</pre>
<pre class="code-block">
tracking:
  enabled: true
  backend: "wandb"
  wandb:
    project: "steganalysis"
</pre>

<hr>

<h2>🔄 Preserving Existing Pipeline</h2>
<ul>
<li><code>train.py</code> — Original SRNet training with pair-constraint batches</li>
<li><code>train_alaska2.py</code> — ALASKA2-specific training loop</li>
<li><code>evaluate_confusion_matrix.py</code> — Standalone confusion matrix evaluation</li>
<li><code>app.py</code> — Streamlit web demo</li>
</ul>
<p>The new <code>framework/</code> module is entirely additive and does not modify any existing code.</p>

<hr>

<h2>📁 Output Structure</h2>
<pre class="code-block">
outputs/
├── resolved_config.yaml        # Exact config used (for reproducibility)
├── test_metrics.json           # Full test set metrics
├── plots/
│   ├── loss_curve.png
│   ├── auc_curves.png
│   ├── f1_curves.png
│   └── lr_schedule.png
├── evaluation_results/
│   ├── roc_curve.png
│   ├── pr_curve.png
│   └── confusion_matrix.png
└── robustness_results/
    ├── robustness_results.json
    ├── robustness_jpeg_compression.png
    └── robustness_gaussian_noise.png
</pre>

</body>
</html>
