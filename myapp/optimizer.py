
import os
import contextlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

class Optimizer:
    def __init__(self, df_person_settings: pd.DataFrame, df_task_settings: pd.DataFrame, suitability_matrix: pd.DataFrame, threshold: float):
        self.df_person_settings = df_person_settings
        self.df_task_settings = df_task_settings
        self.suitability_matrix = Optimizer.scale_scores(suitability_matrix)
        self.threshold = threshold  # 閾値をインスタンス変数として保存
        
        zero_matrix = np.zeros(
            (df_person_settings.shape[0], df_task_settings.shape[0])
        )

        self.df_fix = pd.DataFrame(zero_matrix,
                                   index=self.person_full_name(),
                                   columns=df_task_settings["task_name"])

        self._sr_n_person = df_task_settings["n_people"]
        self.prohibited_tasks = {}

    
    def scale_scores(matrix):
        # スコアの範囲を0から10の間に調整
        min_score = np.min(matrix)
        max_score = np.max(matrix)
        scaled_matrix = 10 * (matrix - min_score) / (max_score - min_score)
        return scaled_matrix
        
        
    def prohibit_task(self, person_full_name: str, task_name: str):
        if person_full_name in self.prohibited_tasks:
            self.prohibited_tasks[person_full_name].append(task_name)
        else:
            self.prohibited_tasks[person_full_name] = [task_name]

    def _auto_assignments(self) -> List[Tuple[int, int]]:
        nonzero = np.nonzero(self.df_fix.values == -1)
        return list(zip(*nonzero))

    def _specific_assignments(self) -> List[Tuple[int, int]]:
        nonzero = np.nonzero(self.df_fix.values > 0)
        return list(zip(*nonzero))

    def person_number(self, full_name: str) -> int:
        return self.df_fix.index.get_loc(full_name)

    def task_number(self, task_name: str) -> int:
        return self.df_fix.columns.get_loc(task_name)

    def person_full_name(self) -> pd.Series:
        last_name = self.df_person_settings["last_name"]
        first_name = self.df_person_settings["first_name"]
        return last_name + " " + first_name

    def sr_n_person(self):
        n_specific = np.count_nonzero(self.df_fix.values > 0, axis=0)
        return self.df_task_settings["n_people"] - n_specific

    def sr_task_hour_per_person(self):
        sr_n_person = self.sr_n_person()

        return (
            self.df_task_settings["hour"]
            .subtract(self.df_fix.replace(-1, 0).sum(axis=0).values)
            .divide(sr_n_person)
            .replace([np.inf, -np.inf, np.nan], 0.0)
        )

    def sr_expected_hour(self):
        subtrahend_hour = np.zeros(self.df_fix.shape[0])

        for e in range(self.df_fix.shape[0]):
            hour = 0.0
            for p in range(self.df_fix.shape[1]):
                fixed = self.df_fix.iloc[e, p]

                if fixed == 0:
                    continue
                elif fixed == -1:
                    sr_task_hour = self.sr_task_hour_per_person()
                    sr_n_person = self.sr_n_person()

                    hour += float(
                        sr_task_hour[p] / sr_n_person[p])
                else:
                    hour += fixed

            subtrahend_hour[e] = hour

        return (
            self.df_person_settings["hour"]
            .subtract(subtrahend_hour)
        )

    def adjust_n_person(self, task_name: str, n: int):
        task_number = self.task_number(task_name)
        self._sr_n_person.at[task_number] = n

    def fix_condition(self,
                      person_full_name: str,
                      task_name: str,
                      hour: float):

        task_number = self.task_number(task_name)

        if self.sr_n_person()[task_number] == 0:
            raise Exception("number of person exceeded the limit")

        if hour > 0.0:
            self.df_fix.loc[person_full_name, task_name] = hour
        elif hour == 0.0:
            self.df_fix.loc[person_full_name, task_name] = -1
        else:
            raise Exception("unsupported hour")

    def optimize(self,
                 sr_expected_hour: pd.Series,
                 sr_task_hour_per_person: pd.Series,
                 sr_n_person: pd.Series,
                 auto_assignments: List[Tuple[int, int]],
                 specific_asignments: List[Tuple[int, int]]):

        n_people = len(sr_expected_hour)
        n_task = len(sr_task_hour_per_person)
        # Log the shapes for debugging
        print(f"Shape of suitability_matrix: {self.suitability_matrix.shape}")
        print(f"Length of sr_task_hour_per_person: {len(sr_task_hour_per_person)}")
        
        # 適性マトリクスの各列にタスクの時間を掛ける
        weighted_suitability = -self.suitability_matrix * sr_task_hour_per_person.values[np.newaxis, :]
        
        c = np.concatenate((weighted_suitability.values.flatten(), np.ones(n_people)))

        # 適性マトリクスの閾値を設定（ここでは中央値を使用）
        # threshold = np.median(self.suitability_matrix)

        # 閾値をインスタンス変数から取得
        threshold = self.threshold

#        # Objective function
#        c = np.concatenate((np.zeros(n_people*n_task), np.ones(n_people)))

        # Constraints
        A_eq = []
        b_eq = []

        for i in range(self.suitability_matrix.shape[0]):
            for j in range(self.suitability_matrix.shape[1]):
                if self.suitability_matrix.iloc[i, j] < threshold:
                    A_row = np.zeros(n_people * n_task + n_people)
                    A_row[i * n_task + j] = 1
                    A_eq.append(A_row)
                    b_eq.append(0)

        # Constraint: Sum of tasks assigned to each person
        for i in range(n_people):
            A_row = np.zeros(n_people*n_task + n_people)
            A_row[i*n_task:(i+1)*n_task] = sr_task_hour_per_person
            A_row[n_people*n_task + i] = -1
            A_eq.append(A_row)
            b_eq.append(sr_expected_hour[i])

        # Constraint: Each task is assigned to sr_n_person people
        for i, n in enumerate(sr_n_person):
            A_row = np.zeros(n_people*n_task + n_people)
            A_row[i::n_task] = 1
            A_eq.append(A_row)
            b_eq.append(n)

        # Additional constraints for auto_assignments and specific_asignments
        for coord in auto_assignments:
            A_row = np.zeros(n_people*n_task + n_people)
            A_row[coord[0]*n_task + coord[1]] = 1
            A_eq.append(A_row)
            b_eq.append(1)

        for coord in specific_asignments:
            A_row = np.zeros(n_people*n_task + n_people)
            A_row[coord[0]*n_task + coord[1]] = 1
            A_eq.append(A_row)
            b_eq.append(0)

        # Constraints for prohibited tasks
        for person, tasks in self.prohibited_tasks.items():
            person_idx = self.df_fix.index.get_loc(person)
            for task in tasks:
                task_idx = self.df_fix.columns.get_loc(task)
                A_row = np.zeros(n_people*n_task + n_people)
                A_row[person_idx*n_task + task_idx] = 1
                A_eq.append(A_row)
                b_eq.append(0)

        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)

        # Bounds
        bounds = [(0, 1) for _ in range(n_people*n_task)] + [(0, None) for _ in range(n_people)]

        # Solve the linear programming problem
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='simplex')

        xs = np.array(res.x[:n_people*n_task]).reshape(n_people, n_task)

        df_out = pd.DataFrame(xs)
        return df_out * self.sr_task_hour_per_person()

    def exec(self) -> pd.DataFrame:
        sr_expected_hour = self.sr_expected_hour()
        sr_task_hour_per_person = self.sr_task_hour_per_person()
        sr_n_person = self.sr_n_person()

        df = self.optimize(
            sr_expected_hour=sr_expected_hour,
            sr_task_hour_per_person=sr_task_hour_per_person,
            sr_n_person=sr_n_person,
            auto_assignments=self._auto_assignments(),
            specific_asignments=self._specific_assignments(),
        )

        df.index = self.person_full_name()
        df.columns = self.df_task_settings["task_name"]

        for e in range(self.df_fix.shape[0]):
            for p in range(self.df_fix.shape[1]):
                fixed = self.df_fix.iloc[e, p]

                hour = 0.0
                if fixed == 0:
                    continue
                elif fixed == -1:
                    sr_task_hour = self.sr_task_hour_per_person()
                    sr_n_person = self.sr_n_person()

                    hour += float(
                        sr_task_hour[p] / sr_n_person[p])
                else:
                    hour += fixed

                df.iloc[e, p] = hour

        df.loc[:, "SUM"] = df.sum(axis=1)
        df.loc["SUM", :] = df.sum(axis=0)

        return df

    def update_task_progress(self, task_name: str, progress_percentage: int):
            task_hours = self.df_task_settings.loc[self.df_task_settings["task_name"] == task_name, "hour"].values[0]
            updated_hours = task_hours * (1 - (progress_percentage / 100))
            self.df_task_settings.loc[self.df_task_settings["task_name"] == task_name, "hour"] = updated_hours

    def calculate_categorical_match(employee: pd.Series, task: pd.Series) -> float:
        match_score = 0
        if employee['Department Encoded'] == task['Required Department Encoded']:
            match_score += 1
        if abs(employee['Career Level Encoded'] - task['Priority Encoded']) <= 1:
            match_score += 1
        return match_score / 2

    def calculate_suitability_matrix(employee_df: pd.DataFrame, task_df: pd.DataFrame) -> pd.DataFrame:
        # Create skill vectors for employees and tasks
        employee_skill_vectors = employee_df[['Skill Set', 'Communication', 'Project Management', 'Coding']].values
        task_skill_vectors = task_df[['Skill Set', 'Communication', 'Project Management', 'Coding']].values

        # Create and fit label encoders
        career_level_encoder = LabelEncoder().fit(list(employee_df['Career Level']) + list(task_df['Priority']))
        department_encoder = LabelEncoder().fit(list(employee_df['Department']) + list(task_df['Required Department']))

        # Transform the categorical data
        employee_df['Career Level Encoded'] = career_level_encoder.transform(employee_df['Career Level'])
        employee_df['Department Encoded'] = department_encoder.transform(employee_df['Department'])
        task_df['Priority Encoded'] = career_level_encoder.transform(task_df['Priority'])
        task_df['Required Department Encoded'] = department_encoder.transform(task_df['Required Department'])

        # Compute the cosine similarity between each employee and task skill vector
        skill_similarity_matrix = cosine_similarity(employee_skill_vectors, task_skill_vectors)

        # Initialize a DataFrame to hold the combined fitness scores
        combined_fitness_scores_df = pd.DataFrame(skill_similarity_matrix,
                                                  index=employee_df['first_name'] + ' ' + employee_df['last_name'],
                                                  columns=task_df['task_name'])

        # Calculate the combined fitness scores
        for task_index, task in task_df.iterrows():
            for employee_index, employee in employee_df.iterrows():
                categorical_match_score = calculate_categorical_match(employee, task)
                combined_fitness_score = (combined_fitness_scores_df.iloc[employee_index, task_index] + categorical_match_score) / 2
                combined_fitness_scores_df.iloc[employee_index, task_index] = combined_fitness_score

        return combined_fitness_scores_df

    