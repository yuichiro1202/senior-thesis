
from optimizer import Optimizer
from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'some_secret_key'

optimizer = None
persons = []
tasks = []

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

def calculate_categorical_match(employee: pd.Series, task: pd.Series) -> float:
    match_score = 0
    if employee['Department Encoded'] == task['Required Department Encoded']:
        match_score += 1
    if abs(employee['Career Level Encoded'] - task['Priority Encoded']) <= 1:
        match_score += 1
    return match_score / 2

@app.route('/upload', methods=['POST'])
def upload():
    global optimizer, persons, tasks

    # ファイルの存在チェック
    if 'persons_file' not in request.files or 'tasks_file' not in request.files:
        return "Missing files.", 400

    persons_file = request.files['persons_file']
    tasks_file = request.files['tasks_file']

    # ファイル選択チェック
    if persons_file.filename == '' or tasks_file.filename == '':
        return "No selected file.", 400

    # ファイルの読み込み
    df_persons = pd.read_csv(persons_file, encoding='cp932')
    df_tasks = pd.read_csv(tasks_file, encoding='cp932')

    # # ユーザーが提供した閾値を取得（フォームから提供されると仮定）
    # threshold = float(request.form.get('threshold', 0.5))  # デフォルト値は 0.5

    # # 適合度行列の計算
    # suitability_matrix = calculate_suitability_matrix(df_persons, df_tasks)

    # 適合度行列と中央値の計算
    suitability_matrix = calculate_suitability_matrix(df_persons, df_tasks)
    median_threshold = suitability_matrix.median().median()

    # セッションに中央値を保存
    session['median_threshold'] = median_threshold

    # Optimizerの初期化
    # optimizer = Optimizer(df_persons, df_tasks, suitability_matrix, threshold)

    optimizer = Optimizer(df_persons, df_tasks, suitability_matrix, median_threshold)

    # セッションと変数の更新
    persons = df_persons.to_dict(orient='records')
    tasks = df_tasks.to_dict(orient='records')
    session['persons'] = persons

    session['initial_persons'] = df_persons.to_dict(orient='records')
    session['initial_tasks'] = df_tasks.to_dict(orient='records')


    # 設定ページへのリダイレクト
    return redirect(url_for('settings'))


@app.route('/use_sample_csvs', methods=['POST'])
def use_sample_csvs():
    global optimizer, persons, tasks
    
    # CSVファイルを読み込む
    df_persons = pd.read_csv(f'{os.getcwd()}/updated_employee_settings.csv',encoding='cp932')
    df_tasks = pd.read_csv(f'{os.getcwd()}/updated_task_settings.csv',encoding='cp932')
#
#    df_persons = pd.read_csv(f'{os.getcwd()}/myapp/example_employee.csv',encoding='utf-8_sig')
#    df_tasks = pd.read_csv(f'{os.getcwd()}/myapp/example_task.csv',encoding='utf-8_sig')
    
    # グローバル変数を更新
    persons = df_persons.to_dict(orient='records')
    tasks = df_tasks.to_dict(orient='records')
    session['persons'] = persons  # Also store it in session
    
    # 適合度行列と中央値の計算
    suitability_matrix = calculate_suitability_matrix(df_persons, df_tasks)
    median_threshold = suitability_matrix.median().median()

    # セッションに中央値を保存
    session['median_threshold'] = median_threshold
    # オプティマイザを初期化
    optimizer = Optimizer(df_persons, df_tasks, suitability_matrix, median_threshold)
    
    # セッションと変数の更新
    persons = df_persons.to_dict(orient='records')
    tasks = df_tasks.to_dict(orient='records')
    session['persons'] = persons

    session['initial_persons'] = df_persons.to_dict(orient='records')
    session['initial_tasks'] = df_tasks.to_dict(orient='records')
    # 設定ページにリダイレクト
    return redirect(url_for('settings'))


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    global optimizer

    if optimizer is None:
        return "Optimizer not initialized. First upload data.", 400

    if request.method == 'POST':
        # threshold = float(request.form.get('threshold', 0.5))  # 閾値をフォームから取得
        # threshold = float(request.form.get('threshold', session.get('median_threshold', 0.5)))
        threshold_str = request.form.get('threshold', str(session.get('median_threshold', 0.5)))

        # 空文字列の場合は0.0に、そうでなければfloatに変換
        if threshold_str == '':
            threshold = 0.0
        else:
            try:
                threshold = float(threshold_str)
            except ValueError:
                # 数値に変換できない場合はデフォルト値0.5を使用
                threshold = 0.5
        

        for person in session['persons']:
            person_name = f"{person['last_name']} {person['first_name']}"
            fixed_tasks = request.form.getlist(f"{person['last_name']}_{person['first_name']}_fixed_tasks")
            prohibited_tasks = request.form.getlist(f"{person['last_name']}_{person['first_name']}_prohibited_tasks")
            
            for task in fixed_tasks:
                hour_input_name = f"{person['last_name']}_{person['first_name']}_{task}_fixed_hours"
                if hour_input_name in request.form:  # チェックされた固定タスクの時間を取得
                    hour = float(request.form[hour_input_name])
                    if hour > 0:
                        optimizer.fix_condition(person_name, task, hour)  # 指定された時間で固定タスクを設定
                    else:
                        optimizer.fix_condition(person_name, task, 0)  # 0は自動割り当てを意味します
            for task in prohibited_tasks:
                optimizer.prohibit_task(person_name, task)

        for task in tasks:
            min_people = request.form.get(f"{task['task_name']}_min_people", 1)
            optimizer.adjust_n_person(task['task_name'], int(min_people))
            

        # 適合度行列と閾値の再設定
        df_persons = pd.DataFrame(persons)
        df_tasks = pd.DataFrame(tasks)
        suitability_matrix = calculate_suitability_matrix(df_persons, df_tasks)
        optimizer = Optimizer(df_persons, df_tasks, suitability_matrix, threshold)
        return redirect(url_for('results'))

    # 初期閾値をセッションから取得
    
    tasks_with_n_people = [{"task_name": task["task_name"], "n_people": optimizer.df_task_settings.loc[optimizer.df_task_settings["task_name"] == task["task_name"], "n_people"].iloc[0]} for task in tasks]
    initial_threshold = "{:.2f}".format(session.get('median_threshold', 0.5))
    # return render_template('settings.html', persons=persons, tasks=tasks_with_n_people, threshold=optimizer.threshold)
    return render_template('settings.html', persons=persons, tasks=tasks_with_n_people, threshold=initial_threshold)

@app.route('/results', methods=['GET', 'POST'])
def results():
    global optimizer

    if optimizer is None:
        return "Optimizer not initialized. First upload data.", 400

    optimized_data = optimizer.exec()
    optimized_data = optimized_data.round(1)  # Round to 1 decimal place
    optimized_dict = optimized_data.to_dict()

    # ヒートマップのデータを作成
    heatmap_df = optimized_data.iloc[:-1, :-1]  # "SUM"行と列を除外
    z = heatmap_df.values.tolist()  # ndarrayをリストに変換
    x = heatmap_df.columns.tolist()
    y = heatmap_df.index.tolist()

    # 結果と期待の差を計算
    names = [f"{person['last_name']} {person['first_name']}" for person in session['persons']]
    sr_diff = optimized_data.iloc[:-1, -1].values - optimizer.df_person_settings["hour"]
    # 名前とその差分のデータを辞書として作成
    diff = dict(zip(names, sr_diff))

    # Get the first person's name
    first_person = next(iter(optimized_dict.keys()))
    
    optimized_data = optimizer.exec()
    
    employee_df = pd.DataFrame(persons)
    task_df = pd.DataFrame(tasks)
    # Include suitability matrix in the results
    suitability_matrix = calculate_suitability_matrix(employee_df, task_df)

    return render_template('results.html', data=optimized_dict, first_person=first_person, heatmap_z=z, heatmap_x=x, heatmap_y=y, diff=diff,tasks=tasks, optimized_data=optimized_data, suitability_matrix=suitability_matrix)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')
    
@app.route('/view_sample_csvs', methods=['GET', 'POST'])
def view_sample_csvs():
    # サンプルCSVファイルを読み込む
    df_persons = pd.read_csv(f'{os.getcwd()}/updated_employee_settings.csv',encoding='utf-8_sig')
    df_tasks = pd.read_csv(f'{os.getcwd()}/updated_task_settings.csv',encoding='utf-8_sig')
    
    # テンプレートにデータを渡してレンダリング
    return render_template('view_sample_csvs.html', persons_df=df_persons, tasks_df=df_tasks)

from flask import request, jsonify

# ここ
# @app.route('/recalculate', methods=['POST'])
# def recalculate():
#     global optimizer

#     if optimizer is None:
#         return jsonify({"error": "Optimizer not initialized"}), 400

#     progress_data = request.json

#     # 進捗データに基づいてタスクを更新
#     for task_name, progress in progress_data.items():
#         # ここで進捗データを使用してoptimizerを更新する
#         optimizer.update_task_progress(task_name, int(progress))

#     # 再計算を実行
#     optimizer.exec()

#     return jsonify({"success": "Recalculated successfully"}), 200

@app.route('/recalculate', methods=['POST'])
def recalculate():
    global optimizer

    if optimizer is None:
        return jsonify({"error": "Optimizer not initialized"}), 400

    # オプティマイザーを初期状態にリセット
    df_persons_initial = pd.DataFrame(session.get('initial_persons', []))
    df_tasks_initial = pd.DataFrame(session.get('initial_tasks', []))
    if df_persons_initial.empty or df_tasks_initial.empty:
        return jsonify({"error": "Initial data not found"}), 400

    suitability_matrix = calculate_suitability_matrix(df_persons_initial, df_tasks_initial)
    optimizer = Optimizer(df_persons_initial, df_tasks_initial, suitability_matrix, session.get('median_threshold', 0.5))

    # 進捗更新を適用
    progress_data = request.json
    for task_name, progress in progress_data.items():
        optimizer.update_task_progress(task_name, int(progress))

    # 最適化の実行
    optimizer.exec()

    return jsonify({"success": "Recalculated successfully"}), 200





if __name__ == "__main__":
    app.run(debug=True)

# from optimizer import Optimizer
# from flask import Flask, request, render_template, redirect, url_for, session
# import pandas as pd
# import os
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics.pairwise import cosine_similarity

# app = Flask(__name__)
# app.secret_key = 'some_secret_key'

# optimizer = None
# persons = []
# tasks = []

# def calculate_suitability_matrix(employee_df: pd.DataFrame, task_df: pd.DataFrame) -> pd.DataFrame:
#     # Create skill vectors for employees and tasks
#     employee_skill_vectors = employee_df[['Skill Set', 'Communication', 'Project Management', 'Coding']].values
#     task_skill_vectors = task_df[['Skill Set', 'Communication', 'Project Management', 'Coding']].values

#     # Create and fit label encoders
#     career_level_encoder = LabelEncoder().fit(list(employee_df['Career Level']) + list(task_df['Priority']))
#     department_encoder = LabelEncoder().fit(list(employee_df['Department']) + list(task_df['Required Department']))
    
#     # Transform the categorical data
#     employee_df['Career Level Encoded'] = career_level_encoder.transform(employee_df['Career Level'])
#     employee_df['Department Encoded'] = department_encoder.transform(employee_df['Department'])
#     task_df['Priority Encoded'] = career_level_encoder.transform(task_df['Priority'])
#     task_df['Required Department Encoded'] = department_encoder.transform(task_df['Required Department'])

#     # Compute the cosine similarity between each employee and task skill vector
#     skill_similarity_matrix = cosine_similarity(employee_skill_vectors, task_skill_vectors)

#     # Initialize a DataFrame to hold the combined fitness scores
#     combined_fitness_scores_df = pd.DataFrame(skill_similarity_matrix,
#                                               index=employee_df['first_name'] + ' ' + employee_df['last_name'],
#                                               columns=task_df['task_name'])

#     # Calculate the combined fitness scores
#     for task_index, task in task_df.iterrows():
#         for employee_index, employee in employee_df.iterrows():
#             categorical_match_score = calculate_categorical_match(employee, task)
#             combined_fitness_score = (combined_fitness_scores_df.iloc[employee_index, task_index] + categorical_match_score) / 2
#             combined_fitness_scores_df.iloc[employee_index, task_index] = combined_fitness_score

#     return combined_fitness_scores_df

# def calculate_categorical_match(employee: pd.Series, task: pd.Series) -> float:
#     match_score = 0
#     if employee['Department Encoded'] == task['Required Department Encoded']:
#         match_score += 1
#     if abs(employee['Career Level Encoded'] - task['Priority Encoded']) <= 1:
#         match_score += 1
#     return match_score / 2

# @app.route('/upload', methods=['POST'])
# def upload():
#     global optimizer, persons, tasks

#     # ファイルの存在チェック
#     if 'persons_file' not in request.files or 'tasks_file' not in request.files:
#         return "Missing files.", 400

#     persons_file = request.files['persons_file']
#     tasks_file = request.files['tasks_file']

#     # ファイル選択チェック
#     if persons_file.filename == '' or tasks_file.filename == '':
#         return "No selected file.", 400

#     # ファイルの読み込み
#     df_persons = pd.read_csv(persons_file, encoding='cp932')
#     df_tasks = pd.read_csv(tasks_file, encoding='cp932')

#     # # ユーザーが提供した閾値を取得（フォームから提供されると仮定）
#     # threshold = float(request.form.get('threshold', 0.5))  # デフォルト値は 0.5

#     # # 適合度行列の計算
#     # suitability_matrix = calculate_suitability_matrix(df_persons, df_tasks)

#     # 適合度行列と中央値の計算
#     suitability_matrix = calculate_suitability_matrix(df_persons, df_tasks)
#     median_threshold = suitability_matrix.median().median()

#     # セッションに中央値を保存
#     session['median_threshold'] = median_threshold

#     # Optimizerの初期化
#     # optimizer = Optimizer(df_persons, df_tasks, suitability_matrix, threshold)

#     optimizer = Optimizer(df_persons, df_tasks, suitability_matrix, median_threshold)

#     # セッションと変数の更新
#     persons = df_persons.to_dict(orient='records')
#     tasks = df_tasks.to_dict(orient='records')
#     session['persons'] = persons

#     session['initial_persons'] = df_persons.to_dict(orient='records')
#     session['initial_tasks'] = df_tasks.to_dict(orient='records')


#     # 設定ページへのリダイレクト
#     return redirect(url_for('settings'))


# @app.route('/use_sample_csvs', methods=['POST'])
# def use_sample_csvs():
#     global optimizer, persons, tasks
    
#     # CSVファイルを読み込む
#     df_persons = pd.read_csv(f'{os.getcwd()}/updated_employee_settings.csv',encoding='cp932')
#     df_tasks = pd.read_csv(f'{os.getcwd()}/updated_task_settings.csv',encoding='cp932')
    
#     # グローバル変数を更新
#     persons = df_persons.to_dict(orient='records')
#     tasks = df_tasks.to_dict(orient='records')
#     session['persons'] = persons  # Also store it in session
    
#     # 適合度行列と中央値の計算
#     suitability_matrix = calculate_suitability_matrix(df_persons, df_tasks)
#     median_threshold = suitability_matrix.median().median()

#     # セッションに中央値を保存
#     session['median_threshold'] = median_threshold
#     # オプティマイザを初期化
#     optimizer = Optimizer(df_persons, df_tasks, suitability_matrix, median_threshold)
    
#     # セッションと変数の更新
#     persons = df_persons.to_dict(orient='records')
#     tasks = df_tasks.to_dict(orient='records')
#     session['persons'] = persons

#     session['initial_persons'] = df_persons.to_dict(orient='records')
#     session['initial_tasks'] = df_tasks.to_dict(orient='records')
#     # 設定ページにリダイレクト
#     return redirect(url_for('settings'))


# @app.route('/settings', methods=['GET', 'POST'])
# def settings():
#     global optimizer

#     if optimizer is None:
#         return "Optimizer not initialized. First upload data.", 400

#     if request.method == 'POST':
#         # threshold = float(request.form.get('threshold', 0.5))  # 閾値をフォームから取得
#         # threshold = float(request.form.get('threshold', session.get('median_threshold', 0.5)))
#         threshold_str = request.form.get('threshold', str(session.get('median_threshold', 0.5)))

#         # 空文字列の場合は0.0に、そうでなければfloatに変換
#         if threshold_str == '':
#             threshold = 0.0
#         else:
#             try:
#                 threshold = float(threshold_str)
#             except ValueError:
#                 # 数値に変換できない場合はデフォルト値0.5を使用
#                 threshold = 0.5
        

#         for person in session['persons']:
#             person_name = f"{person['last_name']} {person['first_name']}"
#             fixed_tasks = request.form.getlist(f"{person['last_name']}_{person['first_name']}_fixed_tasks")
#             prohibited_tasks = request.form.getlist(f"{person['last_name']}_{person['first_name']}_prohibited_tasks")
            
#             for task in fixed_tasks:
#                 hour_input_name = f"{person['last_name']}_{person['first_name']}_{task}_fixed_hours"
#                 if hour_input_name in request.form:  # チェックされた固定タスクの時間を取得
#                     hour = float(request.form[hour_input_name])
#                     if hour > 0:
#                         optimizer.fix_condition(person_name, task, hour)  # 指定された時間で固定タスクを設定
#                     else:
#                         optimizer.fix_condition(person_name, task, 0)  # 0は自動割り当てを意味します
#             for task in prohibited_tasks:
#                 optimizer.prohibit_task(person_name, task)

#         for task in tasks:
#             min_people = request.form.get(f"{task['task_name']}_min_people", 1)
#             optimizer.adjust_n_person(task['task_name'], int(min_people))
            

#         # 適合度行列と閾値の再設定
#         df_persons = pd.DataFrame(persons)
#         df_tasks = pd.DataFrame(tasks)
#         suitability_matrix = calculate_suitability_matrix(df_persons, df_tasks)
#         optimizer = Optimizer(df_persons, df_tasks, suitability_matrix, threshold)
#         return redirect(url_for('results'))

#     # 初期閾値をセッションから取得
    
#     tasks_with_n_people = [{"task_name": task["task_name"], "n_people": optimizer.df_task_settings.loc[optimizer.df_task_settings["task_name"] == task["task_name"], "n_people"].iloc[0]} for task in tasks]
#     initial_threshold = "{:.2f}".format(session.get('median_threshold', 0.5))
#     # return render_template('settings.html', persons=persons, tasks=tasks_with_n_people, threshold=optimizer.threshold)
#     return render_template('settings.html', persons=persons, tasks=tasks_with_n_people, threshold=initial_threshold)

# @app.route('/results', methods=['GET', 'POST'])
# def results():
#     global optimizer

#     if optimizer is None:
#         return "Optimizer not initialized. First upload data.", 400

#     optimized_data = optimizer.exec()
#     optimized_data = optimized_data.round(1)  # Round to 1 decimal place
#     optimized_dict = optimized_data.to_dict()

#     # ヒートマップのデータを作成
#     heatmap_df = optimized_data.iloc[:-1, :-1]  # "SUM"行と列を除外
#     z = heatmap_df.values.tolist()  # ndarrayをリストに変換
#     x = heatmap_df.columns.tolist()
#     y = heatmap_df.index.tolist()

#     # 結果と期待の差を計算
#     names = [f"{person['last_name']} {person['first_name']}" for person in session['persons']]
#     sr_diff = optimized_data.iloc[:-1, -1].values - optimizer.df_person_settings["hour"]
#     # 名前とその差分のデータを辞書として作成
#     diff = dict(zip(names, sr_diff))

#     # Get the first person's name
#     first_person = next(iter(optimized_dict.keys()))
    
#     optimized_data = optimizer.exec()
    
#     employee_df = pd.DataFrame(persons)
#     task_df = pd.DataFrame(tasks)
#     # Include suitability matrix in the results
#     suitability_matrix = calculate_suitability_matrix(employee_df, task_df)

#     return render_template('results.html', data=optimized_dict, first_person=first_person, heatmap_z=z, heatmap_x=x, heatmap_y=y, diff=diff,tasks=tasks, optimized_data=optimized_data, suitability_matrix=suitability_matrix)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     return render_template('index.html')
    
# @app.route('/view_sample_csvs', methods=['GET', 'POST'])
# def view_sample_csvs():
#     # サンプルCSVファイルを読み込む
#     df_persons = pd.read_csv(f'{os.getcwd()}/updated_employee_settings.csv',encoding='utf-8_sig')
#     df_tasks = pd.read_csv(f'{os.getcwd()}/updated_task_settings.csv',encoding='utf-8_sig')
    
#     # テンプレートにデータを渡してレンダリング
#     return render_template('view_sample_csvs.html', persons_df=df_persons, tasks_df=df_tasks)

# from flask import request, jsonify

# @app.route('/recalculate', methods=['POST'])
# def recalculate():
#     global optimizer

#     if optimizer is None:
#         return jsonify({"error": "Optimizer not initialized"}), 400

#     # オプティマイザーを初期状態にリセット
#     df_persons_initial = pd.DataFrame(session.get('initial_persons', []))
#     df_tasks_initial = pd.DataFrame(session.get('initial_tasks', []))
#     if df_persons_initial.empty or df_tasks_initial.empty:
#         return jsonify({"error": "Initial data not found"}), 400

#     suitability_matrix = calculate_suitability_matrix(df_persons_initial, df_tasks_initial)
#     optimizer = Optimizer(df_persons_initial, df_tasks_initial, suitability_matrix, session.get('median_threshold', 0.5))

#     # 進捗更新を適用
#     progress_data = request.json
#     for task_name, progress in progress_data.items():
#         optimizer.update_task_progress(task_name, int(progress))

#     # 最適化の実行
#     optimizer.exec()

#     return jsonify({"success": "Recalculated successfully"}), 200





# if __name__ == "__main__":
#     app.run(debug=True)
