import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
import streamlit as st

st.set_page_config(page_title="医疗费用预测", page_icon='💰')
pd.set_option("display.unicode.east_asian_width", True)

# 尝试加载数据，添加错误处理
try:
    insurance_df = pd.read_csv("insurance-chinese.csv", encoding='gbk')
    output = insurance_df['医疗费用']
    features = insurance_df[['年龄', '性别', 'BMI', '子女数量', '是否吸烟', '区域']]
    features = pd.get_dummies(features)
    
    print(features.head())
    print(output.head())

    x_train, x_test, y_train, y_test = train_test_split(features, output, train_size=0.8)
    rfr = RandomForestRegressor()
    rfr.fit(x_train, y_train)
    y_pred = rfr.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print(f'该模型的可决系数（R-squared)是:{r2}')

    with open('rfr_model.pkl', 'wb') as f:
        pickle.dump(rfr, f)
    print('保存成功，已生成相关文件。')
    
except FileNotFoundError:
    st.error("找不到数据文件，请检查文件路径是否正确。")
    st.stop()
except Exception as e:
    st.error(f"数据加载或模型训练出错: {e}")
    st.stop()

def introduce_page():
    st.write('# 欢迎使用！')
    st.sidebar.success('单击👈🏼预测医疗费用')
    st.markdown(
        """
        # 医疗费用预测费用💰
        这个应用利用机器学习模型来预测医疗费用，为保险公司的保险定价提供参考。
        ## 背景介绍
        - 开发目标：帮助保险公司合理定价保险产品，控制风险。
        - 模型算法：利用随机森林回归算法训练医疗费用预测模型。
        ## 使用指南
        - 输入准确完整的被保险人信息，可以得到更准确的费用预测。
        - 预测结果可以作为保险定价的重要参考，但需审慎决策。
        - 有任何问题欢迎联系我们的技术支持。
        技术支持：email：support@example.com
        """
    )

def predict_page():
    st.markdown(
        """
        ## 使用说明
        这个应用利用机器学习模型来预测医疗费用，为保险公司的保险定价提供参考。
        -.**输入信息**：在下面输入被保险人的个人信息、疾病信息等。
        -.**费用预测**：应用会预测被保险人的未来医疗费用支出。
        """
    )
    
    with st.form('user_inputs'):
        age = st.number_input('年龄', min_value=0)
        sex = st.radio('性别', options=['男性', '女性'])
        bmi = st.number_input('BMI', min_value=0.0)
        children = st.number_input('子女数量：', step=1, min_value=0)
        smoke = st.radio('是否吸烟', ('是', '否'))
        region = st.selectbox('区域', ('东南部', '西南部', '东北部', '西北部'))
        submitted = st.form_submit_button('预测费用')
    
    if submitted:
        st.write('用户输入的数据是：')
        st.text([age, sex, bmi, children, smoke, region])
        
        # 编码分类变量
        sex_female, sex_male = 0, 0
        if sex == '女性':
            sex_female = 1
        elif sex == '男性':
            sex_male = 1
            
        smoke_yes, smoke_no = 0, 0
        if smoke == '是':
            smoke_yes = 1
        elif smoke == '否':
            smoke_no = 1
            
        region_northeast, region_southeast, region_northwest, region_southwest = 0, 0, 0, 0
        if region == '东北部':
            region_northeast = 1
        elif region == '东南部':
            region_southeast = 1
        elif region == '西北部':
            region_northwest = 1
        elif region == '西南部':
            region_southwest = 1
        
        format_data = [age, bmi, children, sex_female, sex_male, smoke_no, smoke_yes, 
                       region_northeast, region_southeast, region_northwest, region_southwest]
        
        # 加载模型并预测
        try:
            with open('rfr_model.pkl', 'rb') as f:
                rfr_model = pickle.load(f)
            
            format_data_df = pd.DataFrame(data=[format_data], columns=rfr_model.feature_names_in_)
            predict_result = rfr_model.predict(format_data_df)[0]
            st.write('根据您输入的数据，预测该顾客的医疗费用是：', round(predict_result, 2))
        except Exception as e:
            st.error(f"预测出错: {e}")
    
    st.write("技术支持：email：support@example.com")

# 导航逻辑移到主程序
nav = st.sidebar.radio("导航", ["简介", "预测医疗费用"])
if nav == "简介":
    introduce_page()
else:
    predict_page()
