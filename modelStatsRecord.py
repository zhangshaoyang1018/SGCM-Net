# -*- coding: utf-8 -*-
import numpy as np
import time
import collections
from sklearn import metrics
from datetime import datetime
#import averageAccuracy
# ===================================================================================================================UP数据集用到的输出打印格式
# def outputRecord(ELEMENT_ACC_RES_SS4, AA_RES_SS4, OA_RES_SS4, KAPPA_RES_SS4,
#                  ELEMENT_PRE_RES_SS4, AP_RES_SS4, TRAINING_TIME_RES_SS4, TESTING_TIME_RES_SS4,
#                  CATEGORY, ITER, path1):
#     print_matrix = np.zeros((CATEGORY * 2 + 6, ITER + 1), dtype=object)
#     print_matrix[0:CATEGORY, 0:ITER] = np.around(ELEMENT_ACC_RES_SS4, 4)
#     print_matrix[CATEGORY, 0:ITER] = np.around(AA_RES_SS4, 4)
#     print_matrix[CATEGORY + 1, 0:ITER] = np.around(OA_RES_SS4, 4)
#     print_matrix[CATEGORY + 2, 0:ITER] = np.around(KAPPA_RES_SS4, 4)
#     print_matrix[CATEGORY + 3:CATEGORY * 2 + 3, 0:ITER] = np.around(ELEMENT_PRE_RES_SS4, 4)
#     print_matrix[CATEGORY * 2 + 3, 0:ITER] = np.around(AP_RES_SS4, 4)
#     print_matrix[CATEGORY * 2 + 4, 0:ITER] = np.around(TRAINING_TIME_RES_SS4, 4)
#     print_matrix[CATEGORY * 2 + 5, 0:ITER] = np.around(TESTING_TIME_RES_SS4, 4)
#     element_mean = np.mean(print_matrix[:, :-1], axis=1)
#     element_std = np.std(np.float64(print_matrix[:, :-1]), axis=1)
#     for i in range(CATEGORY * 2 + 4):
#         print_matrix[i, ITER] = "{:.2f}".format(element_mean[i] * 100) + " ± " + "{:.2f}".format(element_std[i] * 100)
#     for i in range((CATEGORY * 2 + 4), (CATEGORY * 2 + 6)):
#         print_matrix[i, ITER] = "{:.2f}".format(element_mean[i]) + " ± " + "{:.2f}".format(element_std[i])
#     np.savetxt(path1, print_matrix.astype(str), fmt='%s', delimiter="\t", newline='\n')
def outputRecord_noP(
    ELEMENT_ACC,      # shape: (C, ITER)  每类Recall/PA
    AA,               # shape: (ITER,)     每次实验AA
    OA,               # shape: (ITER,) or (1,ITER) 每次实验OA
    KAPPA,            # shape: (ITER,) or (1,ITER) 每次实验Kappa
    TRAIN_TIME,       # shape: (ITER,) or (1,ITER)
    TEST_TIME,        # shape: (ITER,) or (1,ITER)
    CATEGORY,         # C
    ITER,             # 实验次数
    path1,            # 输出txt路径
    meta=None,        # dict: 额外元信息（可选）
    class_names=None, # list[str] 长度C（可选）
):
    """
    输出更可读的实验记录：
    - 去掉 P/AP（Precision相关）
    - 增加元信息header
    - 增加行名/列名
    """

    # --------- 0) 规范shape：把 (1,ITER) 压成 (ITER,) ----------
    def _to_1d(x):
        x = np.asarray(x)
        if x.ndim == 2 and x.shape[0] == 1:
            return x.reshape(-1)
        if x.ndim == 2 and x.shape[1] == 1:
            return x.reshape(-1)
        return x

    AA = _to_1d(AA)
    OA = _to_1d(OA)
    KAPPA = _to_1d(KAPPA)
    TRAIN_TIME = _to_1d(TRAIN_TIME)
    TEST_TIME = _to_1d(TEST_TIME)

    ELEMENT_ACC = np.asarray(ELEMENT_ACC)  # (C, ITER)

    assert ELEMENT_ACC.shape == (CATEGORY, ITER), f"ELEMENT_ACC shape should be {(CATEGORY, ITER)}, got {ELEMENT_ACC.shape}"
    assert AA.shape[0] == ITER and OA.shape[0] == ITER and KAPPA.shape[0] == ITER, "AA/OA/KAPPA length must equal ITER"
    assert TRAIN_TIME.shape[0] == ITER and TEST_TIME.shape[0] == ITER, "TRAIN/TEST time length must equal ITER"

    # --------- 1) 行名：每类PA + AA/OA/Kappa + time ----------
    if class_names is None:
        row_names = [f"PA/Recall_Class_{i+1}" for i in range(CATEGORY)]
    else:
        assert len(class_names) == CATEGORY
        row_names = [f"PA/Recall_{name}" for name in class_names]

    row_names += ["AA (mean PA/Recall)", "OA (overall accuracy)", "Kappa (Cohen)", "TrainTime (s)", "TestTime (s)"]

    # 行数：C + 5
    n_rows = CATEGORY + 5
    # 列名：exp_1 ... exp_ITER + mean±std
    col_names = [f"exp_{i+1}" for i in range(ITER)] + ["mean±std"]

    # --------- 2) 组装数值矩阵（先用float，最后再转str） ----------
    # data_matrix shape: (n_rows, ITER)
    data_matrix = np.zeros((n_rows, ITER), dtype=np.float64)
    data_matrix[0:CATEGORY, :] = np.around(ELEMENT_ACC, 4)
    data_matrix[CATEGORY, :] = np.around(AA, 4)
    data_matrix[CATEGORY+1, :] = np.around(OA, 4)
    data_matrix[CATEGORY+2, :] = np.around(KAPPA, 4)
    data_matrix[CATEGORY+3, :] = np.around(TRAIN_TIME, 4)
    data_matrix[CATEGORY+4, :] = np.around(TEST_TIME, 4)

    # --------- 3) 计算 mean/std ----------
    mean = data_matrix.mean(axis=1)
    std = data_matrix.std(axis=1) if ITER > 1 else np.full((n_rows,), np.nan)

    # --------- 4) 构造最终可打印矩阵（带行列名） ----------
    # final shape: (n_rows+1, ITER+2)
    # 第一行做列名，第一列做行名
    final = np.empty((n_rows + 1, ITER + 2), dtype=object)
    final[0, 0] = "Metric"
    final[0, 1:ITER+1] = col_names[:-1]
    final[0, ITER+1] = col_names[-1]

    # 填入每行的实验值
    for r in range(n_rows):
        final[r+1, 0] = row_names[r]
        final[r+1, 1:ITER+1] = [f"{v:.4f}" for v in data_matrix[r, :]]

        # mean±std：前 (C+3) 行是比例指标（乘100），时间不乘100
        is_ratio = (r < CATEGORY + 3)  # PA/AA/OA/Kappa 都算比例；Train/Test time 不算
        if ITER > 1:
            if is_ratio:
                final[r+1, ITER+1] = f"{mean[r]*100:.2f} ± {std[r]*100:.2f}"
            else:
                final[r+1, ITER+1] = f"{mean[r]:.2f} ± {std[r]:.2f}"
        else:
            # ITER=1 不建议写 std=0.00，会误导你做了多次实验
            if is_ratio:
                final[r+1, ITER+1] = f"{mean[r]*100:.2f} ± N/A"
            else:
                final[r+1, ITER+1] = f"{mean[r]:.2f} ± N/A"

    # --------- 5) 写文件：先写元信息header，再写表格 ----------
    header_lines = []
    header_lines.append("# Experiment Record (no Precision/P)")
    header_lines.append(f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    header_lines.append(f"# CATEGORY(classes)={CATEGORY}, ITER(runs)={ITER}")
    header_lines.append("# Metrics:")
    header_lines.append("# - PA/Recall per class = diag(C) / sum(C, axis=1)")
    header_lines.append("# - AA = mean(PA/Recall over classes)")
    header_lines.append("# - OA = correct/total")
    header_lines.append("# - Kappa = Cohen's kappa")
    header_lines.append("# - Train/Test time in seconds")
    if meta:
        header_lines.append("# Meta:")
        for k, v in meta.items():
            header_lines.append(f"#   {k}: {v}")

    with open(path1, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n")
        # 表格部分用tab分隔
        for r in range(final.shape[0]):
            f.write("\t".join(map(str, final[r, :])) + "\n")
    header_lines.append(f"# EndTime at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
def outputRecord(element_acc, aa, oa, kappa, element_pre, ap,
                 training_time, testing_time, classes_num, iter_num, filename):
    """
    保存实验结果到文件

    参数:
        element_acc: 每类准确率 [classes_num, iter_num]
        aa: 平均准确率 [iter_num]
        oa:  总体准确率 [iter_num, 1]
        kappa:  Kappa系数 [iter_num, 1]
        element_pre: 每类精确率 [classes_num, iter_num]
        ap: 平均精确率 [iter_num]
        training_time: 训练时间 [iter_num, 1]
        testing_time: 测试时间 [iter_num, 1]
        classes_num: 类别数
        iter_num: 迭代次数
        filename: 保存文件名
    """

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENTAL RESULTS - FEW-SHOT HYPERSPECTRAL IMAGE CLASSIFICATION\n")
        f.write("=" * 80 + "\n\n")

        # ========================================
        # 基本信息
        # ========================================
        f.write(f"Number of runs: {iter_num}\n")
        f.write(f"Number of classes: {classes_num}\n\n")

        # ========================================
        # 总体指标
        # ========================================
        f.write("=" * 80 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("=" * 80 + "\n\n")

        if iter_num > 1:
            # 多次实验：显示均值±标准差
            f.write(f"Overall Accuracy (OA):\n")
            f.write(f"  Mean:   {oa.mean() * 100:.2f}%\n")
            f.write(f"  Std:   {oa.std() * 100:.2f}%\n")
            f.write(f"  95% CI: [{(oa.mean() - 1.96 * oa.std() / np.sqrt(iter_num)) * 100:.2f}%, "
                    f"{(oa.mean() + 1.96 * oa.std() / np.sqrt(iter_num)) * 100:.2f}%]\n\n")

            f.write(f"Average Accuracy (AA):\n")
            f.write(f"  Mean:  {aa.mean() * 100:.2f}%\n")
            f.write(f"  Std:    {aa.std() * 100:.2f}%\n")
            f.write(f"  95% CI: [{(aa.mean() - 1.96 * aa.std() / np.sqrt(iter_num)) * 100:.2f}%, "
                    f"{(aa.mean() + 1.96 * aa.std() / np.sqrt(iter_num)) * 100:.2f}%]\n\n")

            f.write(f"Kappa Coefficient:\n")
            f.write(f"  Mean:  {kappa.mean():.4f}\n")
            f.write(f"  Std:    {kappa.std():.4f}\n")
            f.write(f"  95% CI: [{kappa.mean() - 1.96 * kappa.std() / np.sqrt(iter_num):.4f}, "
                    f"{kappa.mean() + 1.96 * kappa.std() / np.sqrt(iter_num):.4f}]\n\n")
        else:
            # 单次实验：只显示数值
            f.write(f"Overall Accuracy (OA):     {oa[0, 0] * 100:.2f}%\n")
            f.write(f"Average Accuracy (AA):     {aa[0] * 100:.2f}%\n")
            f.write(f"Kappa Coefficient:         {kappa[0, 0]:.4f}\n\n")

        # ========================================
        # 每类准确率
        # ========================================
        f.write("=" * 80 + "\n")
        f.write("PER-CLASS ACCURACY\n")
        f.write("=" * 80 + "\n\n")

        if iter_num > 1:
            f.write(f"{'Class':<8} {'Mean (%)':<12} {'Std (%)':<12} {'Min (%)':<12} {'Max (%)':<12}\n")
            f.write("-" * 80 + "\n")
            for i in range(classes_num):
                f.write(f"{i + 1:<8} {element_acc[i, :].mean() * 100:<12.2f} "
                        f"{element_acc[i, :].std() * 100:<12.2f} "
                        f"{element_acc[i, :].min() * 100:<12.2f} "
                        f"{element_acc[i, :].max() * 100:<12.2f}\n")
        else:
            f.write(f"{'Class':<8} {'Accuracy (%)':<15}\n")
            f.write("-" * 80 + "\n")
            for i in range(classes_num):
                f.write(f"{i + 1:<8} {element_acc[i, 0] * 100:<15.2f}\n")

        f.write("\n")

        # ========================================
        # 每类精确率
        # ========================================
        f.write("=" * 80 + "\n")
        f.write("PER-CLASS PRECISION\n")
        f.write("=" * 80 + "\n\n")

        if iter_num > 1:
            f.write(f"{'Class':<8} {'Mean (%)':<12} {'Std (%)':<12} {'Min (%)':<12} {'Max (%)':<12}\n")
            f.write("-" * 80 + "\n")
            for i in range(classes_num):
                f.write(f"{i + 1:<8} {element_pre[i, :].mean() * 100:<12.2f} "
                        f"{element_pre[i, :].std() * 100:<12.2f} "
                        f"{element_pre[i, :].min() * 100:<12.2f} "
                        f"{element_pre[i, :].max() * 100:<12.2f}\n")
        else:
            f.write(f"{'Class':<8} {'Precision (%)':<15}\n")
            f.write("-" * 80 + "\n")
            for i in range(classes_num):
                f.write(f"{i + 1:<8} {element_pre[i, 0] * 100:<15.2f}\n")

        f.write("\n")

        # ========================================
        # 时间统计
        # ========================================
        f.write("=" * 80 + "\n")
        f.write("TIME STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        if iter_num > 1:
            f.write(f"Training Time:\n")
            f.write(f"  Mean:  {training_time.mean():.2f} seconds\n")
            f.write(f"  Std:   {training_time.std():.2f} seconds\n")
            f.write(f"  Total: {training_time.sum():.2f} seconds\n\n")

            f.write(f"Testing Time:\n")
            f.write(f"  Mean:  {testing_time.mean():.2f} seconds\n")
            f.write(f"  Std:    {testing_time.std():.2f} seconds\n")
            f.write(f"  Total: {testing_time.sum():.2f} seconds\n\n")
        else:
            f.write(f"Training Time:   {training_time[0, 0]:.2f} seconds\n")
            f.write(f"Testing Time:   {testing_time[0, 0]:.2f} seconds\n\n")

        # ========================================
        # 详细结果（每次运行）
        # ========================================
        if iter_num > 1:
            f.write("=" * 80 + "\n")
            f.write("DETAILED RESULTS FOR EACH RUN\n")
            f.write("=" * 80 + "\n\n")

            for run in range(iter_num):
                f.write(f"--- Run {run + 1}/{iter_num} ---\n")
                f.write(f"OA:    {oa[run, 0] * 100:.2f}%\n")
                f.write(f"AA:    {aa[run] * 100:.2f}%\n")
                f.write(f"Kappa: {kappa[run, 0]:.4f}\n")
                f.write(f"Time:  {training_time[run, 0]:.2f}s (train) + {testing_time[run, 0]:.2f}s (test)\n")
                f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"\n✓ Results saved to:  {filename}")

def outputStats(KAPPA_AE, OA_AE, AA_AE, ELEMENT_ACC_AE, TRAINING_TIME_AE, TESTING_TIME_AE, history, loss_and_metrics, CATEGORY, path1, path2):


    f = open(path1, 'a')

    sentence0 = 'KAPPAs, mean_KAPPA ± std_KAPPA for each iteration are:' + str(KAPPA_AE) + str(np.mean(KAPPA_AE)) + ' ± ' + str(np.std(KAPPA_AE)) + '\n'
    f.write(sentence0)
    sentence1 = 'OAs, mean_OA ± std_OA for each iteration are:' + str(OA_AE) + str(np.mean(OA_AE)) + ' ± ' + str(np.std(OA_AE)) + '\n'
    f.write(sentence1)
    sentence2 = 'AAs, mean_AA ± std_AA for each iteration are:' + str(AA_AE) + str(np.mean(AA_AE)) + ' ± ' + str(np.std(AA_AE)) + '\n'
    f.write(sentence2)
    sentence3 = 'Total average Training time is :' + str(np.sum(TRAINING_TIME_AE)) + '\n'
    f.write(sentence3)
    sentence4 = 'Total average Testing time is:' + str(np.sum(TESTING_TIME_AE)) + '\n'
    f.write(sentence4)

    element_mean = np.mean(ELEMENT_ACC_AE, axis=0)
    element_std = np.std(ELEMENT_ACC_AE, axis=0)
    sentence5 = "Mean of all elements in confusion matrix:" + str(np.mean(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence5)
    sentence6 = "Standard deviation of all elements in confusion matrix" + str(np.std(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence6)

    f.close()

    print_matrix = np.zeros((CATEGORY), dtype=object)
    for i in range(CATEGORY):
        print_matrix[i] = str(element_mean[i]) + " ± " + str(element_std[i])

    np.savetxt(path2, print_matrix.astype(str), fmt='%s', delimiter="\t",
               newline='\n')

    print('Test score:', loss_and_metrics[0])
    print('Test accuracy:', loss_and_metrics[1])
    print(history.history.keys())


def outputStats_assess(KAPPA_AE, OA_AE, AA_AE, ELEMENT_ACC_AE, CATEGORY, path1, path2):


    f = open(path1, 'a')

    sentence0 = 'KAPPAs, mean_KAPPA ± std_KAPPA for each iteration are:' + str(KAPPA_AE) + str(np.mean(KAPPA_AE)) + ' ± ' + str(np.std(KAPPA_AE)) + '\n'
    f.write(sentence0)
    sentence1 = 'OAs, mean_OA ± std_OA for each iteration are:' + str(OA_AE) + str(np.mean(OA_AE)) + ' ± ' + str(np.std(OA_AE)) + '\n'
    f.write(sentence1)
    sentence2 = 'AAs, mean_AA ± std_AA for each iteration are:' + str(AA_AE) + str(np.mean(AA_AE)) + ' ± ' + str(np.std(AA_AE)) + '\n'
    f.write(sentence2)

    element_mean = np.mean(ELEMENT_ACC_AE, axis=0)
    element_std = np.std(ELEMENT_ACC_AE, axis=0)
    sentence5 = "Mean of all elements in confusion matrix:" + str(np.mean(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence5)
    sentence6 = "Standard deviation of all elements in confusion matrix" + str(np.std(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence6)

    f.close()

    print_matrix = np.zeros((CATEGORY), dtype=object)
    for i in range(CATEGORY):
        print_matrix[i] = str(element_mean[i]) + " ± " + str(element_std[i])

    np.savetxt(path2, print_matrix.astype(str), fmt='%s', delimiter="\t",
               newline='\n')


def outputStats_SVM(KAPPA_AE, OA_AE, AA_AE, ELEMENT_ACC_AE, TRAINING_TIME_AE, TESTING_TIME_AE, CATEGORY, path1, path2):


    f = open(path1, 'a')

    sentence0 = 'KAPPAs, mean_KAPPA ± std_KAPPA for each iteration are:' + str(KAPPA_AE) + str(np.mean(KAPPA_AE)) + ' ± ' + str(np.std(KAPPA_AE)) + '\n'
    f.write(sentence0)
    sentence1 = 'OAs, mean_OA ± std_OA for each iteration are:' + str(OA_AE) + str(np.mean(OA_AE)) + ' ± ' + str(np.std(OA_AE)) + '\n'
    f.write(sentence1)
    sentence2 = 'AAs, mean_AA ± std_AA for each iteration are:' + str(AA_AE) + str(np.mean(AA_AE)) + ' ± ' + str(np.std(AA_AE)) + '\n'
    f.write(sentence2)
    sentence3 = 'Total average Training time is :' + str(np.sum(TRAINING_TIME_AE)) + '\n'
    f.write(sentence3)
    sentence4 = 'Total average Testing time is:' + str(np.sum(TESTING_TIME_AE)) + '\n'
    f.write(sentence4)

    element_mean = np.mean(ELEMENT_ACC_AE, axis=0)
    element_std = np.std(ELEMENT_ACC_AE, axis=0)
    sentence5 = "Mean of all elements in confusion matrix:" + str(np.mean(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence5)
    sentence6 = "Standard deviation of all elements in confusion matrix" + str(np.std(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence6)

    f.close()

    print_matrix = np.zeros((CATEGORY), dtype=object)
    for i in range(CATEGORY):
        print_matrix[i] = str(element_mean[i]) + " ± " + str(element_std[i])

    np.savetxt(path2, print_matrix.astype(str), fmt='%s', delimiter="\t",
               newline='\n')