import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class Tsk:
    def __init__(self):
        self.features = None
        self.df = pd.read_csv('creditcard.csv')

    def data(self):
        self.df = self.df.drop(['Time'], axis=1)
        self.df = self.df.drop_duplicates()
        print(self.df['Class'].value_counts())
        print(self.df.head(10))
        self.df.info()
        print(self.df.describe())
        print("Назви колонок:", self.df.columns.tolist())

    def ft_imp(self):
        X = self.df.drop(['Class'], axis=1)
        y = self.df['Class']

        self.X_tain, self.X_test, self.y_tarin, self.y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=33)

        clf = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=33, class_weight='balanced')
        clf.fit(self.X_tain, self.y_tarin)

        imptances = clf.feature_importances_
        self.feat_imp = pd.DataFrame({'feature': X.columns, 'importance': imptances})
        self.feat_imp = self.feat_imp.sort_values('importance', ascending=False).reset_index(drop=True)
        self.features = ['V10', 'V14', 'V4', 'V12', 'V11', 'V17', 'Class']
        self.df = self.df[self.features].copy(deep=True)
        self.df.info


    def cls_vs_feat(self):
        fig, axes = plt.subplots(3, 2, figsize=(17, 12))
        fig.suptitle('Features vs class', size = 18)

        for ax, feat in zip(axes.flatten(), self.features):
            sns.boxplot(
                ax=ax,
                data=self.df,
                x='Class',
                y=feat,
                hue='Class',
                palette='Spectral',
                dodge=False
            )
            ax.set_title(f"{feat} distribution")
            ax.get_legend().remove()
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 8))
        fig.suptitle('Destributions\n', size=18)
        for ax, feat in zip(axes.flatten(), self.features):
            ax.hist(self.df[feat], bins=60, linewidth=0.5, edgecolor='green')
            ax.set_title(f"{feat} distribution")
        plt.tight_layout()

    def visual_ft_imp(self):
        print(self.feat_imp)
        plt.figure(figsize=(5,10))
        plt.barh(self.feat_imp['feature'], self.feat_imp['importance'])
        plt.gca().invert_yaxis()
        plt.xlabel('Importance')
        plt.title('Feature')
        plt.tight_layout()
        plt.show()

    def distributions(self):
        self.features = ['V10', 'V14', 'V4', 'V11', 'V17']
        sns.pairplot(self.df[self.features], diag_kind='hist', plot_kws={'alpha':0.4, 's':20})
        plt.suptitle('Pairwise Scatterplots', y=1.02)
        plt.show()

    def IQR_method(self, n=2, k=1.5, visualize=False):
        from collections import Counter
        features = self.features[:-1]
        df = self.df.copy()
        outlier_list = []

        for column in features:
            Q1 = np.percentile(df[column], 25)
            Q3 = np.percentile(df[column], 75)
            IQR = Q3 - Q1
            outlier_step = k * IQR

            outlier_indices = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step)].index
            outlier_list.extend(outlier_indices)

        outlier_counter = Counter(outlier_list)
        multiple_outliers = [idx for idx, count in outlier_counter.items() if count > n]

        print(f"Знайдено {len(multiple_outliers)} рядків з >{n} викидами")

        if visualize:
            df_out = df[features].copy()
            df_out['anomaly'] = 'normal'
            df_out.loc[multiple_outliers, 'anomaly'] = 'outlier'

            sns.pairplot(
                df_out,
                vars=features,
                hue='anomaly',
                palette={'normal': 'blue', 'outlier': 'red'},
                diag_kind='hist',
                plot_kws={'alpha': 0.4, 's': 20}
            )
            plt.suptitle('Pairwise Scatterplots with IQR Outliers Highlighted', y=1.02)
            plt.show()

            for group, title in [('normal', 'Normal Observations'), ('outlier', 'IQR Outliers')]:
                group_df = df_out[df_out['anomaly'] == group]
                fig, axes = plt.subplots(3, 2, figsize=(12, 8))
                fig.suptitle(f'Feature Distributions: {title}', size=16)

                for ax, feat in zip(axes.flatten(), features):
                    ax.hist(group_df[feat], bins=60, alpha=0.7, edgecolor='white')
                    ax.set_title(f"{feat} ({group})")
                    ax.set_xlabel(feat)
                    ax.set_ylabel("Count")

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()

            # Оцінка точності
            df_out['pred_iqr'] = 0
            df_out.loc[multiple_outliers, 'pred_iqr'] = 1

            y_true = df['Class']
            y_pred = df_out['pred_iqr']

            from sklearn.metrics import classification_report, confusion_matrix
            print("=== IQR Outlier Detection vs. True Fraud Labels ===")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=['non‑fraud', 'fraud']))

            cm = confusion_matrix(y_true, y_pred)
            cm_df_out = pd.DataFrame(cm,
                                     index=['true_non‑fraud', 'true_fraud'],
                                     columns=['pred_non‑fraud', 'pred_fraud'])
            print("\nConfusion Matrix:")
            print(cm_df_out)

        return multiple_outliers


start_tsk = Tsk()
start_tsk.data()
start_tsk.ft_imp()
start_tsk.visual_ft_imp()
start_tsk.cls_vs_feat()
start_tsk.distributions()
outlier_ids = start_tsk.IQR_method()
print(f"Індекси викидів: {outlier_ids}")
print(f"Знайдено {len(outlier_ids)} рядків з >2 викидами")