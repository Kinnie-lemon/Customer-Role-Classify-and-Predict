import pandas as pd
import numpy as np
from jieba import lcut
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import tkinter as tk
from tkinter import filedialog, dialog
import os
import pickle


def predict(params, p_type):
    # 如果是传进来excel
    # df_pred=pd.read_excel('Online sales customer1819.xlsx','result',usecols=[1,13])
    # 如果是传进来文本，以下
    # Name = '上海电气集团股份有限公司'
    # Pwd = '电站及输配电,机电一体化,交通运输、环保设备的相关装备制造业产品的设计、制造、销售,提供相关'
    if p_type == 'one':  # 传单个公司，对应左边的功能
        Name, Pwd = params
        df_pred = pd.DataFrame({'Query.Name': [Name], 'Result.Scope': [Pwd]})
    elif p_type == 'many':
        df_pred = pd.read_excel(
            file_path)  # (r'D:\Company Role Cleansing\01 Raw Data\01_1st batch_QCC result(cleaned).xlsx')
        df_pred.columns = ['Query.Name', 'Result.Scope']
    elif p_type == 'default':
        df_full = pd.read_excel(r'D:\Company Role Cleansing\01 Raw Data\01_1st batch_QCC result(cleaned).xlsx')  # 40185
        df_full = df_full[pd.notnull(df_full['经营范围'])]  # 40177
        df_full = df_full.reset_index(drop=True)
        df_pred = df_full[['Query.Name', '经营范围']]
        df_pred.rename({'经营范围': 'Result.Scope'}, inplace=True, axis=1)
    else:
        print("你没有说要用哪种预测方法。")
    # 至此，得到df_pred，一个两列的DataFrame
    f = open('tfidf_model.pkl', 'rb')
    (tfidf_model, words_combined) = pickle.load(f)
    f.close()
    f = open('geo_list.pkl', 'rb')
    geo_list = pickle.load(f)
    f.close()
    f = open('industry_list.pkl', 'rb')
    industry_list = pickle.load(f)
    f.close()

    def remove_words(item, word_list):
        filtered_item = item
        for word in word_list:
            filtered_item = filtered_item.replace(word, '')
        return filtered_item

    def extract_word(splited_item, word_list):
        l = []
        for i in splited_item:
            if i in word_list:
                l.append(i)
        return l

    def stopwordslist(filepath):
        stopwords = [str(line).strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
        return stopwords

    def remove_bracket(substr: str):
        def find_next_1(index_left: int, index_right: int):
            index_left = str.find(substr, '(', index_left)  # 第一次出现的左括号
            index_right = str.find(substr, ')', index_right)  # 第二次出现的位置
            return index_left, index_right

        def find_next_2(index_left: int, index_right: int):
            index_left = str.find(substr, '【', index_left)  # 第一次出现的左括号
            index_right = str.find(substr, '】', index_right)  # 第二次出现的位置
            return index_left, index_right

        index_l = 0
        index_r = 0

        while index_l >= 0 & index_r >= 0:
            index_l, index_r = find_next_1(index_l, index_r)
            if index_r > index_l:
                substr = substr[:index_l] + substr[index_r + 1:]
            elif index_r < index_l:

                index_l, index_r = find_next_1(index_l, index_r + 1)
                if index_r < index_l:
                    break
        index_l = 0
        index_r = 0

        while index_l >= 0 & index_r >= 0:
            index_l, index_r = find_next_2(index_l, index_r)
            if index_r > index_l:
                substr = substr[:index_l] + substr[index_r + 1:]
            elif index_r < index_l:

                index_l, index_r = find_next_2(index_l, index_r + 1)
                if index_r < index_l:
                    break
        return substr

    def preprocess_chinese(sentence, letters=False, digital=False):
        """
        该函数可以根据具体要求具体添加，这里默认只保留汉字，可以通过传参
        是否保留英文字母和数字等，也可以在此基础上继续添加特定的功能
        :param sentence:
        :param letters: 默认为False，不保留
        :param digital: 默认为False，不保留
        :return: 处理运行字符串
        """
        if letters & digital:
            pro_str = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
        elif ~letters & digital:
            pro_str = re.compile('[^0-9^\u4e00-\u9fa5]')
        elif letters & ~digital:
            pro_str = re.compile('[^A-Z^a-z^\u4e00-\u9fa5]')
        else:
            pro_str = re.compile('[^\u4e00-\u9fa5]')
        return pro_str.sub('', sentence)

    stopwords_list = stopwordslist('中文停用词表.txt')
    stopwords_list = [stopword.strip() for stopword in stopwords_list]  # 1894
    # 停用词
    stopwords_2 = ['省', '市', '县', '中国',
                   '（', '）', '(', ')', '-', '*', ' ', '·', '.', '?']
    # 标志词
    suffixes = ['有限', '股份', '集团', '分公司', '公司', '厂', '局']
    inds_1 = []
    companies_list = df_pred['Query.Name'].values
    for i in range(len(companies_list)):

        company_name_1_splitted = [i for i in lcut(companies_list[i], HMM=True)]
        geo_extracted_1 = extract_word(company_name_1_splitted, geo_list)
        company_name_1 = remove_words(companies_list[i], geo_extracted_1)
        company_name_2 = remove_words(company_name_1, stopwords_2)
        min_idx = len(company_name_2)
        for suffix in suffixes:
            try:
                suffix_idx = company_name_2.index(suffix)
            except:
                suffix_idx = min_idx
            if suffix_idx < min_idx:
                min_idx = company_name_2.index(suffix)
        if min_idx <= 0:
            print(companies_list[i] + 'failed')
            inds_1.append('')
            continue
        else:
            company_name_3 = company_name_2[:min_idx]
            company_name_3_splitted = [i for i in lcut(company_name_3, HMM=True)]
            inds_1.append(company_name_3_splitted[-1])
    # df_kw_samples['Industry'] = inds_1
    df_pred['Industry2'] = inds_1
    df_pred['Industry3'] = np.where(df_pred['Industry2'].apply(len) >= 4, df_pred['Industry2'].str[-2:],
                                    df_pred['Industry2'])

    df_pred.drop(columns=['Industry2'], inplace=True)

    words_list = []
    for i in range(len(df_pred)):
        content = df_pred.iloc[i, :]['Result.Scope']
        if pd.isna(content):
            print(content)
            words_list.append('')
            continue
        content = remove_bracket(content)
        content = preprocess_chinese(content, letters=False)
        splitedStr = ''
        words = lcut(content, HMM=True)
        words_final = []
        for word in words:
            if word not in stopwords_list:
                words_final.append(word)
        words_list.append(words_final)

    document_pred = [" ".join(word) for word in words_list]
    pred_result = tfidf_model.transform(document_pred)

    pred_result = pred_result[:, list(words_combined.index)]
    X_pred = pred_result.toarray()
    print(X_pred.shape)
    '''
        以下的五个部分是调取每个已训练模型，然后进行预测的步骤。
        和notebook里训练时的流程一样，先进行数据预处理
            不一样的地方在于：之后直接调pkl模型进行预测
            预测结果保存在df_pred_ec和df_pred_ec_prob内，分别代表0/1值与概率值。
    '''
    # EC
    f = open('ec_model.pkl', 'rb')
    (name_EC, name_EC_excl), (clf_EC, clf_EC_stack) = pickle.load(f)
    f.close()

    EC_name_count = 0
    EC_name_excl_count = 0
    EC_name_count_list = []
    EC_name_excl_count_list = []

    for i in range(len(df_pred.Industry3)):
        EC_name_count = 0
        EC_name_excl_count = 0
        company_name = df_pred.Industry3.values[i]
        for kw in name_EC:
            if kw in company_name:
                EC_name_count = 1
                break
        for kw in name_EC_excl:
            if kw in company_name:
                EC_name_excl_count = 1
                break
        EC_name_count_list.append(EC_name_count)
        EC_name_excl_count_list.append(EC_name_excl_count)

    df_name_features = pd.DataFrame({'EC_Name': EC_name_count_list, 'EC_Name_exclude': EC_name_excl_count_list})
    # df_input_model2 = pd.concat([df_result_from_model1, df_name_features.iloc[df_samples_forEC['Index'],:].reset_index(drop = True)], axis = 1).values

    df_pred['Prediction'] = clf_EC.predict(X_pred)
    df_pred_stack = pd.concat([df_pred[['Prediction']].reset_index(drop=True)
                                  , df_name_features.reset_index(drop=True)
                               ], axis=1).values
    df_pred['Prediction_2'] = clf_EC_stack.predict(df_pred_stack)
    df_pred['Prediction_2_Prob_other'] = clf_EC_stack.predict_proba(df_pred_stack)[:, 0]
    df_pred['Prediction_2_Prob1_EC'] = clf_EC_stack.predict_proba(df_pred_stack)[:, 1]
    df_pred_final = pd.concat(
        [df_pred.reset_index(drop=True), pd.DataFrame(df_pred_stack, columns=['Pred_Model1', 'KW_incl', 'KW_excl'])],
        axis=1).copy(deep=True)
    df_pred_ec = np.where(df_pred_final['KW_incl'] + df_pred_final['KW_excl'] == 0, 1, df_pred_final['Prediction_2'])
    df_pred_ec = pd.Series(df_pred_ec)
    df_pred_ec_prob = df_pred_final['Prediction_2_Prob1_EC']
    df_pred_ec_prob_hidden = pd.Series(np.where(df_pred_ec == 1, df_pred_ec_prob + 0.2, df_pred_ec_prob))

    # 为了避免误导新改的，见下面，放弃原来0/1的设定
    df_pred_ec = df_pred_final['Prediction_2']

    # PaB
    f = open('pab_model.pkl', 'rb')
    (name_EC, name_EC_excl), (clf_PAB, clf_PAB_stack) = pickle.load(f)
    f.close()

    EC_name_count = 0
    EC_name_excl_count = 0
    EC_name_count_list = []
    EC_name_excl_count_list = []

    for i in range(len(df_pred.Industry3)):
        EC_name_count = 0
        EC_name_excl_count = 0
        company_name = df_pred.Industry3.values[i]
        for kw in name_EC:
            if kw in company_name:
                EC_name_count = 1
                break
        for kw in name_EC_excl:
            if kw in company_name:
                EC_name_excl_count = 1
                break
        EC_name_count_list.append(EC_name_count)
        EC_name_excl_count_list.append(EC_name_excl_count)

    range_EC = ['低压', '开关', '配电']
    EC_range_count = 0
    EC_range_count_list = []

    for i in range(len(df_pred['Result.Scope'])):
        EC_range_count = 0
        company_name = df_pred['Result.Scope'].values[i]
        for kw in range_EC:
            try:
                if kw in company_name:
                    EC_range_count = 1
                    break
            except:
                print(company_name)
        EC_range_count_list.append(EC_range_count)

    df_name_features = pd.DataFrame({'EC_Name': EC_name_count_list, 'EC_Name_exclude': EC_name_excl_count_list})
    df_range_features = pd.DataFrame({'EC_range': EC_range_count_list})  ##Ben新加的公司经营范围的特征

    df_pred['Prediction'] = clf_PAB.predict(X_pred)
    df_pred_stack = pd.concat([df_pred[['Prediction']].reset_index(drop=True)
                                  , df_name_features.reset_index(drop=True)
                                  , df_range_features.reset_index(drop=True)
                               ], axis=1).values
    df_pred['Prediction_2'] = clf_PAB_stack.predict(df_pred_stack)
    df_pred['Prediction_2_Prob_other'] = clf_PAB_stack.predict_proba(df_pred_stack)[:, 0]
    df_pred['Prediction_2_Prob1_EC'] = clf_PAB_stack.predict_proba(df_pred_stack)[:, 1]
    df_pred_final = pd.concat([df_pred.reset_index(drop=True),
                               pd.DataFrame(df_pred_stack, columns=['Pred_Model1', 'KW_incl', 'KW_excl', 'Range_KW'])],
                              axis=1).copy(deep=True)

    df_pred_pab = df_pred_final['Prediction_2']
    df_pred_pab.rename(columns={'Final_Pred': 'PaB'}, inplace=True)
    df_pred_pab_prob = df_pred_final['Prediction_2_Prob1_EC']

    # OEM
    f = open('oem_model.pkl', 'rb')
    (name_EC, name_EC_excl), (clf_OEM, clf_OEM_stack) = pickle.load(f)
    f.close()

    EC_name_count = 0
    EC_name_excl_count = 0
    EC_name_count_list = []
    EC_name_excl_count_list = []

    for i in range(len(df_pred.Industry3)):
        EC_name_count = 0
        EC_name_excl_count = 0
        company_name = df_pred.Industry3.values[i]
        for kw in name_EC:
            if kw in company_name:
                EC_name_count = 1
                break
        for kw in name_EC_excl:
            if kw in company_name:
                EC_name_excl_count = 1
                break
        EC_name_count_list.append(EC_name_count)
        EC_name_excl_count_list.append(EC_name_excl_count)



    range_EC = ['机器人', '铸造', '智能']
    EC_range_count = 0
    EC_range_count_list = []

    for i in range(len(df_pred['Result.Scope'])):
        EC_range_count = 0
        company_name = df_pred['Result.Scope'].values[i]
        for kw in range_EC:
            try:
                if kw in company_name:
                    EC_range_count = 1
                    break
            except:
                print(company_name)
        EC_range_count_list.append(EC_range_count)

    df_name_features = pd.DataFrame({'EC_Name': EC_name_count_list, 'EC_Name_exclude': EC_name_excl_count_list})
    df_range_features = pd.DataFrame({'EC_range': EC_range_count_list})  ##Ben新加的公司经营范围的特征

    df_pred['Prediction'] = clf_OEM.predict(X_pred)
    df_pred_stack = pd.concat([df_pred[['Prediction']].reset_index(drop=True)
                                  , df_name_features.reset_index(drop=True)
                                  , df_range_features.reset_index(drop=True)
                               ], axis=1).values
    df_pred['Prediction_2'] = clf_OEM_stack.predict(df_pred_stack)
    df_pred['Prediction_2_Prob_other'] = clf_OEM_stack.predict_proba(df_pred_stack)[:, 0]
    df_pred['Prediction_2_Prob1_EC'] = clf_OEM_stack.predict_proba(df_pred_stack)[:, 1]
    df_pred_final = pd.concat([df_pred.reset_index(drop=True),
                               pd.DataFrame(df_pred_stack, columns=['Pred_Model1', 'KW_incl', 'KW_excl', 'Range_KW'])],
                              axis=1).copy(deep=True)

    df_pred_oem = df_pred_final['Prediction_2']
    df_pred_oem_prob = df_pred_final['Prediction_2_Prob1_EC']

    # SI
    f = open('si_model.pkl', 'rb')
    (name_EC, name_EC_excl), (clf_SI, clf_SI_stack) = pickle.load(f)
    f.close()

    EC_name_count = 0
    EC_name_excl_count = 0
    EC_name_count_list = []
    EC_name_excl_count_list = []

    for i in range(len(df_pred.Industry3)):
        EC_name_count = 0
        EC_name_excl_count = 0
        company_name = df_pred.Industry3.values[i]
        for kw in name_EC:
            if kw in company_name:
                EC_name_count = 1
                break
        for kw in name_EC_excl:
            if kw in company_name:
                EC_name_excl_count = 1
                break
        EC_name_count_list.append(EC_name_count)
        EC_name_excl_count_list.append(EC_name_excl_count)

    range_EC = []  # ['编程', '开发', '计算机', '承包', '系统集成', '软件', '服务', '外包', '系统']
    EC_range_count = 0
    EC_range_count_list = []

    for i in range(len(df_pred['Result.Scope'])):
        EC_range_count = 0
        company_name = df_pred['Result.Scope'].values[i]
        for kw in range_EC:
            try:
                if kw in company_name:
                    EC_range_count = 1
                    break
            except:
                print(company_name)
        EC_range_count_list.append(EC_range_count)

    df_name_features = pd.DataFrame({'EC_Name': EC_name_count_list, 'EC_Name_exclude': EC_name_excl_count_list})
    df_range_features = pd.DataFrame({'EC_range': EC_range_count_list})  # Ben新加的公司经营范围的特征

    df_pred['Prediction'] = clf_SI.predict(X_pred)
    df_pred_stack = pd.concat([df_pred[['Prediction']].reset_index(drop=True)
                                  , df_name_features.reset_index(drop=True)
                                  , df_range_features.reset_index(drop=True)
                               ], axis=1).values
    df_pred['Prediction_2'] = clf_SI_stack.predict(df_pred_stack)
    df_pred['Prediction_2_Prob_other'] = clf_SI_stack.predict_proba(df_pred_stack)[:, 0]
    df_pred['Prediction_2_Prob1_EC'] = clf_SI_stack.predict_proba(df_pred_stack)[:, 1]
    df_pred_final = pd.concat([df_pred.reset_index(drop=True),
                               pd.DataFrame(df_pred_stack, columns=['Pred_Model1', 'KW_incl', 'KW_excl', 'Range_KW'])],
                              axis=1).copy(deep=True)

    df_pred_si = df_pred_final['Prediction_2']
    df_pred_si_prob = df_pred_final['Prediction_2_Prob1_EC']

    # Dist
    f = open('dist_model.pkl', 'rb')
    (name_EC, name_EC_excl), (clf_DIST, clf_DIST_stack) = pickle.load(f)
    f.close()

    EC_name_count = 0
    EC_name_excl_count = 0
    EC_name_count_list = []
    EC_name_excl_count_list = []

    for i in range(len(df_pred.Industry3)):
        EC_name_count = 0
        EC_name_excl_count = 0
        company_name = df_pred.Industry3.values[i]
        for kw in name_EC:
            if kw in company_name:
                EC_name_count = 1
                break
        for kw in name_EC_excl:
            if kw in company_name:
                EC_name_excl_count = 1
                break
        EC_name_count_list.append(EC_name_count)
        EC_name_excl_count_list.append(EC_name_excl_count)

    range_EC = []
    EC_range_count = 0
    EC_range_count_list = []

    for i in range(len(df_pred['Result.Scope'])):
        EC_range_count = 0
        company_name = df_pred['Result.Scope'].values[i]
        for kw in range_EC:
            try:
                if kw in company_name:
                    EC_range_count = 1
                    break
            except:
                print(company_name)
        EC_range_count_list.append(EC_range_count)

    df_name_features = pd.DataFrame({'EC_Name': EC_name_count_list, 'EC_Name_exclude': EC_name_excl_count_list})
    df_range_features = pd.DataFrame({'EC_range': EC_range_count_list})  # Ben新加的公司经营范围的特征

    df_pred['Prediction'] = clf_DIST.predict(X_pred)
    df_pred_stack = pd.concat([df_pred[['Prediction']].reset_index(drop=True)
                                  , df_name_features.reset_index(drop=True)
                                  , df_range_features.reset_index(drop=True)
                               ], axis=1).values
    df_pred['Prediction_2'] = clf_DIST_stack.predict(df_pred_stack)
    df_pred['Prediction_2_Prob_other'] = clf_DIST_stack.predict_proba(df_pred_stack)[:, 0]
    df_pred['Prediction_2_Prob1_EC'] = clf_DIST_stack.predict_proba(df_pred_stack)[:, 1]
    df_pred_final = pd.concat([df_pred.reset_index(drop=True),
                               pd.DataFrame(df_pred_stack, columns=['Pred_Model1', 'KW_incl', 'KW_excl', 'Range_KW'])],
                              axis=1).copy(deep=True)

    df_pred_dist = df_pred_final['Prediction_2']
    df_pred_dist_prob = df_pred_final['Prediction_2_Prob1_EC']

    '''
        从这里开始是整合所有的分类预测0/1结果与分类预测概率结果，并生成最终的导出表。
    '''
    df_pred_pred = pd.concat(
        [df_pred_ec, df_pred_pab, df_pred_oem, df_pred_si, df_pred_dist, df_pred_ec_prob, df_pred_pab_prob,
         df_pred_oem_prob, df_pred_si_prob, df_pred_dist_prob,
         df_pred_ec_prob_hidden],
        axis=1)
    df_pred_pred.columns = ['EC', 'PaB', 'OEM', 'SI', 'Dist', 'EC_prob', 'PaB_prob', 'OEM_prob', 'SI_prob', 'Dist_prob',
                            'EC_hidd']
    # df_pred_pred.index=df_export[~null_idx].index.tolist()
    df_pred_pred['final_pred'] = df_pred_pred[['EC_prob', 'PaB_prob', 'OEM_prob', 'SI_prob', 'Dist_prob']].idxmax(
        axis=1)
    df_pred_pred['final_pred'] = df_pred_pred['final_pred'].apply(lambda x: x[:-5])
    '''
        如果五类的预测概率没有一类大于0.5，就认为是其他，否则选取概率最高的那一类
    '''
    def separate(a, b, c, d, e):
        s = a + b + c + d + e
        if s == 0:
            return 1
        else:
            return 0

    df_pred_pred['is_others'] = df_pred_pred.apply(
        lambda row: separate(row['EC'], row['PaB'], row['OEM'], row['SI'], row['Dist']), axis=1)
    idx_others = df_pred_pred['is_others'] == 1
    df_pred_pred.loc[idx_others, 'final_pred'] = 'others'
    # df_exexport=df_export.join(df_pred_pred)
    df_exexport = pd.concat([df_pred.iloc[:, :2], df_pred_pred.drop(['EC_hidd', 'is_others'], axis=1)], axis=1)

    return df_exexport


from tkinter import StringVar, Label, Entry, Text, Button
import tkinter.messagebox as mb

my_window = tk.Tk()
my_window.title('公司名称&经营范围预测系统——ADBA')
my_window.geometry('400x450')
my_window.geometry('+150+50')

# label 标签（用户名和密码的变量）
varName = StringVar()  # 文字变量储存器器
varName.set("")
# 创建账户的标签
labname = Label(my_window, text="公司名称", justify='right', bg="pink", width=80)
labname.place(x=10, y=5, width=80, height=20)
# 创建文本框，同事设置关联变量
enterName = Entry(my_window, width=80, textvariable=varName)
enterName.place(x=100, y=5, width=250, height=20)
# 创建密码的标签 和文本框
labPwd = Label(my_window, text="经营范围", justify='right', bg="pink", width=80)
labPwd.place(x=10, y=30, width=80, height=20)

enterPwd = Text(my_window, width=80)  # ,textvariable=varPwd)
enterPwd.place(x=100, y=30, width=250, height=100)

# 结果的文本框
labRes = Label(my_window, text="预测概率", justify='right', bg="lightskyblue", width=80)
labRes.place(x=10, y=160, width=80, height=20)

enterRes = Text(my_window, width=50, height=5)  # , textvariable=varRes
enterRes.place(x=100, y=160, width=250, height=100)


# enterRes.config(state='disabled')

# 界面的上半部分：预测单个并输出结果
def predict_single():
    Name = enterName.get()
    Pwd = enterPwd.get('0.0', 'end')
    if Name != "" and Pwd != "":
        print("TBC")
        df_exexport = predict((Name, Pwd), 'one')
        result = df_exexport['final_pred'][0]
        # mb.showinfo(title="分类结果",message=result)
        ec = df_exexport['EC_prob'][0]
        pab = df_exexport['PaB_prob'][0]
        oem = df_exexport['OEM_prob'][0]
        si = df_exexport['SI_prob'][0]
        dist = df_exexport['Dist_prob'][0]
        enterRes.insert("insert",
                        "End Customer\t" + '\t' + '{:.0%}'.format(ec) + "\n" + "PaB\t" + '\t' + '{:.0%}'.format(
                            pab) + "\n" + "OEM\t" + '\t' + '{:.0%}'.format(
                            oem) + "\n" + "SI\t" + '\t' + '{:.0%}'.format(
                            si) + "\n" + "Distributor\t" + '\t' + '{:.0%}'.format(dist) + "\n" * 2)
        enterRes.insert("end", "Final:\t" + result)
    else:
        print("LWB")
        mb.showerror(title="报错", message="输入错误，请重新输入")


# 创建按钮组件，并且设置按钮事件的处理函数
buttonOK = Button(my_window, text="运行", background='lightgray', command=predict_single)
buttonOK.place(x=230, y=135, width=50, height=20)


# 取消按钮的时间处理函数
def reset_text():
    # 清空用户输入的用户名和密码
    varName.set("")
    enterPwd.delete('0.0', "end")
    enterRes.delete('0.0', "end")


buttonCancel = Button(my_window, text="重置", background='lightgray', command=reset_text)
buttonCancel.place(x=300, y=135, width=50, height=20)


def send_to_clibboard():
    my_window.clipboard_clear()
    my_window.clipboard_append(enterRes.get('0.0', 'end'))
    my_window.update()


buttonCopy = Button(my_window, text="复制结果", background='lightgray', command=send_to_clibboard)
buttonCopy.place(x=260, y=270, width=90, height=20)
labPath = Text(my_window, width=50, height=5)  # anchor='nw',
labPath.place(x=100, y=320, width=250, height=40)


def open_file():
    global file_path
    global file_text
    global save_path
    file_path = filedialog.askopenfilename(title=u'选择文件', initialdir=(os.path.expanduser('H:/')))
    # labPath["text"] = file_path
    save_path = file_path[:-5] + 'predicted' + '.xlsx'
    print('打开文件：', file_path)
    labPath.delete('0.0', "end")
    labPath.insert("insert", file_path)
    labPath.insert("end", "")
    '''
    try:
        print("Yeah")
        print(file_path)
        results_excel=predict(file_path,'many')
        results_excel.to_excel(save_path,index=False)
        mb.showinfo(title="处理完成", message='Processed, saved to'+save_path)

    except:
        mb.showerror(title="❌错误", message='No such file' + file_path)
    '''


def batch_predict():  # 执行批量预测
    try:
        print("Yeah")
        print(file_path)
        results_excel = predict(file_path, 'many')
        results_excel.to_excel(save_path, index=False)
        mb.showinfo(title="处理完成", message='Processed, saved to' + save_path)

    except Exception as e:
        print(e)
        mb.showerror(title="❌错误", message='No such file' + file_path)


labPL = Label(my_window, text="批量预测", justify='right', bg="pink", width=80)
labPL.place(x=10, y=320, width=80, height=20)

buttonliulan = tk.Button(my_window, text='浏览', width=15, height=2, background='lightgray', command=open_file)
buttonliulan.place(x=230, y=370, width=50, height=20)

buttonOpen = tk.Button(my_window, text='运行', width=15, height=2, background='lightgray', command=batch_predict)
buttonOpen.place(x=300, y=370, width=50, height=20)

# 进入消息循环
my_window.mainloop()
