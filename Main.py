import numpy as np
import Data
import Model
import Parameter
import csv
import os
import datetime
import time
from tabulate import tabulate
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score

models = ['SRGPIN', 'SGPIN', 'SPIN', 'SH']
metrics = ['accuracies', 'mcc_scores', 'f1_scores']

results = {metric: {model: [] for model in models} for metric in metrics}
times = {model: [] for model in models}

kf = KFold(n_splits=Parameter.num_folds, shuffle=True, random_state=42)

for i in range(1):
    print(f'(Round {i+1})')
    for k, (train_index, test_index) in enumerate(kf.split(Data.X), start=1):
        data = []
        X_train, X_test = Data.X[train_index], Data.X[test_index]
        y_train, y_test = Data.y[train_index], Data.y[test_index]

        classifiers = {
            'SRGPIN': Model.M_BFGS_SRGPIN_SVM(tau1=0.1, tau2=0.9, epsilon1=0.01, epsilon2=0.05, lambd=1, eta=5, mu=0.1, C=1, max_iteration=20),
            'SGPIN': Model.M_BFGS_SGPIN_SVM(tau1=0.7, tau2=0.9, epsilon1=0.05, epsilon2=0.05, mu=0.1, C=1, max_iteration=20),
            'SPIN': Model.M_BFGS_SPIN_SVM(tau=0.9, mu=0.1, C=1, max_iteration=20),
            'SH': Model.M_BFGS_SH_SVM(mu=0.1, C=1, max_iteration=20)
        }

        predictions = {}
        for model, clf in classifiers.items():
            start_time = time.time()
            clf.fit(X_train, y_train)
            times[model].append(time.time() - start_time)
            predictions[model] = clf.predict(X_test)

        row_acc = ['Accuracy']
        row_mcc = ['MCC']
        row_f1 = ['F1 scores']
        for model in models:
            acc = accuracy_score(y_test, predictions[model])
            mcc = matthews_corrcoef(y_test, predictions[model])
            f1 = f1_score(y_test, predictions[model])

            results['accuracies'][model].append(acc)
            results['mcc_scores'][model].append(mcc)
            results['f1_scores'][model].append(f1)

            row_acc.append(f'{acc:.4f}')
            row_mcc.append(f'{mcc:.4f}')
            row_f1.append(f'{f1:.4f}')

        headers = [f'Fold: {k}'] + models
        print(tabulate([row_acc, row_mcc, row_f1], headers=headers, tablefmt="grid"))

summary_data = []
for metric in metrics:
    row = [metric.capitalize().replace('_', ' ')]
    for model in models:
        mean = np.mean(results[metric][model]) * 100
        std = np.std(results[metric][model]) * 100
        row.append(f'{mean:.2f} (±S.D. {std:.2f})')
    summary_data.append(row)

print(tabulate(summary_data, headers=['Average'] + models, tablefmt="grid"))

# Save CSV in the current working directory
file_name = os.path.splitext(os.path.basename(Data.data_file_path))[0]
current_datetime = datetime.datetime.now().strftime("%d_%m_%Y")
csv_file_path = os.path.join(os.getcwd(), f'{file_name}_{current_datetime}.csv')

with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for model in models:
        writer.writerow([f'{model}, Noise level = {Parameter.noise_level}'])
        for metric in metrics:
            mean = np.mean(results[metric][model]) * 100
            std = np.std(results[metric][model]) * 100
            label = metric.replace('_', ' ').capitalize()
            writer.writerow([f'Average {label}', f'{mean:.2f} ± {std:.2f}'])
        writer.writerow([])

# Print average running times
for model in models:
    avg_time = np.mean(times[model])
    print(f"Average time for {model} SVM: {avg_time:.4f} seconds")
