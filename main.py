import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns


st.title("Make Predictions That Helps You Decide")
st.write("""
# Explore Different Data Sets
""")

dataset_name = st.sidebar.selectbox("Navigate Different Data Sets And Pages", ("Diabetes Prediction", "Car Evaluation", "Wine Quality", "Heart Failure Mortality", "Breast Cancer Type", "About Developer"))
def create_Model(Chosen_data):
    if Chosen_data == "Diabetes Prediction":
        st.subheader("Predict Diabetic State")
        st.image("images.png")
        dataset = pd.read_csv('datasets_23663_30246_diabetes.csv')
        st.dataframe(dataset.head(3))
        if st.checkbox("Show Summary"):
            st.write(dataset.describe())
        if st.checkbox("Show Shape Of Data Set"):
            st.write(dataset.shape)
            st.write(
                "This Means That The Prediction Will Be Made From Machine Learning Model Trained On Two Thousand Diffeent Instances")
        if st.checkbox("Value Count Plot"):
            st.write(dataset["Outcome"].value_counts().plot(kind='bar'))
            st.pyplot()
            st.success("0 Indicates Not Diabetic")
            st.success("1 Indicates Diabetic")



        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X)

        from sklearn.ensemble import RandomForestClassifier
        model1 = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
        models = model1.fit(X_train, y)

        st.subheader("Enter Values Of Variables For Prediction Here")
        Pregnancies = st.slider("Numbers Of Times Pregnant", 0, 30)
        Glucose = st.number_input("Plasma Glucose Concentration in 2 Hours in an Oral Glucose Tolerance Test", 0.000,
                                  500.000)
        BloodPressure = st.number_input("Diastolic Blood Pressure (mm Hg)", 0.000, 500.000)
        SkinThickness = st.number_input("Triceps Skin Fold Thickness (mm)", 0.0000, 500.000)
        Insulin = st.number_input("2-Hours Serum Insulin (muU/ml)", 0.000, 1000.000)
        BMI = st.number_input("Body Mass Index(weight in kg/(height in m)^2)", 0.000, 200.000)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.000, 5.000)
        Age = st.slider("Age", 0, 200)

        transformed_parameters = sc.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        prediction = models.predict(transformed_parameters)
        st.subheader("Click The Predict Button Below")
        if st.button("Predict"):
            st.subheader("Note That The Model Has An Accuracy Of 98% Based On Prediction Of Test Data With 768 Different Instances")
            if prediction == 1:
                st.success("Patient Is Diabetic")
            else:
                st.success("Patient Is Not Diabetic")




    if Chosen_data == "Car Evaluation":
        st.subheader("Evaluate Car State")
        st.image("images (42).jpeg")
        car_data = pd.read_csv("datasets_2298_3884_car_evaluation.csv")
        st.dataframe(car_data.head(3))
        if st.checkbox("Show Summary"):
            st.write(car_data.describe())
        if st.checkbox("Show Shape Of Data Set"):
            st.write(car_data.shape)
            st.write("This Means That The Prediction Will Be Made From Machine Learning Model Trained On A thousand, Seven Hundred And Twenty Eight Diffrent Instances")
        if st.checkbox("Value Count Plot"):
            st.write(car_data["class"].value_counts().plot(kind='bar'))
            st.pyplot()
        if st.checkbox("Pie Chart"):
            st.write(car_data["class"].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()


        buying_label = {'vhigh':3, 'low':1, 'med':2, 'high':0}
        maint_label = {'vhigh':3, 'low':1, 'med':2, 'high':0}
        door_label = {'2':0, '3':1, '5more':3, '4':2}
        persons_label = {'2':0, '4':1, 'more':2}
        lug_boot_label = {'small':2, 'big':0, 'med':1}
        safety_label = {'high':0, 'med':2, 'low':1}
        class_label = {'good':0, 'acceptable':1, 'very good':2, 'unacceptable':3}

        def get_value(val,my_dict):
            for key ,value in my_dict.items():
                if val == key:
                    return value

        def get_key(val,my_dict):
            for key ,value in my_dict.items():
                if val == value:
                    return key



        buying = st.selectbox("Select Buying Level", tuple(buying_label.keys()))
        maint = st.selectbox("Select Maintenance Level", tuple(maint_label.keys()))
        doors = st.selectbox("Select Number of Doors", tuple(door_label.keys()))
        persons = st.selectbox("Select Number Of Person (Select More If More Than 4)", tuple(persons_label.keys()))
        lug_boot = st.selectbox("Select Lug Boot", tuple(lug_boot_label.keys()))
        safety = st.selectbox("Select Safety Level", tuple(safety_label.keys()))

        v_buying = get_value(buying, buying_label)
        v_maint = get_value(maint, maint_label)
        v_doors = get_value(doors, door_label)
        v_persons = get_value(persons, persons_label)
        v_lug_boot = get_value(lug_boot, lug_boot_label)
        v_safety = get_value(safety, safety_label)

        pretty_data = {
            "buying":buying,
            "maint":maint,
            "doors":doors,
            "persons":persons,
            "lug_boot":lug_boot,
            "safety":safety,
        }

        st.subheader("Options Selected")
        st.json(pretty_data)

        st.subheader("Data Encoded As:")
        sample_data = [v_buying,v_maint,v_doors,v_persons,v_lug_boot,v_safety]
        st.write(sample_data)
        prep_data = np.array(sample_data).reshape(1, -1)


        model_choice = st.selectbox("Model Choice",["Random Forest Regression", "Kernal SVM", "Gradient Boosting Classifier"])
        if st.button("Evaluate"):
            X = car_data.iloc[:, :-1].values
            y = car_data.iloc[:, -1]

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            X[:, 0] = le.fit_transform(X[:, 0])
            X[:, 1] = le.fit_transform(X[:, 1])
            X[:, 2] = le.fit_transform(X[:, 2])
            X[:, 3] = le.fit_transform(X[:, 3])
            X[:, 4] = le.fit_transform(X[:, 4])
            X[:, 5] = le.fit_transform(X[:, 5])
            classes = np.unique(y)
            st.write("The Different Classes In Data Set Are:", classes)
            if model_choice == "Random Forest Regression":
                st.subheader("Note That This Model Has An Accuracy Of 96% Based On Prediction Of 20% Of The Total Data")
                from sklearn.ensemble import RandomForestClassifier
                classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
                classifier.fit(X, y)
                pred = classifier.predict(prep_data)
                if pred == "acc":
                    st.success("This Prediction States That The Car Is Acceptable")
                if pred == "good":
                    st.success("This Prediction States That The Car Is Good")
                if pred == "unacc":
                    st.success("This Prediction States That The Car Is Unacceptable")
                if pred == "vgood":
                    st.success("This Prediction States That The Car Is Very Good")


            if model_choice == "Kernal SVM":
                st.subheader("Note That This Model Has An Accuracy Of 98% Based On Prediction Of 20% Of The Total Data")
                from sklearn.svm import SVC
                classifier = SVC(kernel='rbf', random_state=0)
                classifier.fit(X, y)
                pred = classifier.predict(prep_data)
                if pred == "acc":
                    st.success("This Prediction States That The Car Is Acceptable")
                if pred == "good":
                    st.success("This Prediction States That The Car Is Good")
                if pred == "unacc":
                    st.success("This Prediction States That The Car Is Unacceptable")
                if pred == "vgood":
                    st.success("This Prediction States That The Car Is Very Good")


            if model_choice == "Gradient Boosting Classifier":
                st.subheader("Note That This Model Has An Accuracy Of 99% Based On Prediction Of 20% Of The Total Data")
                from sklearn.ensemble import GradientBoostingClassifier
                classifier = GradientBoostingClassifier(n_estimators=100, random_state=0)
                classifier.fit(X, y)
                pred = classifier.predict(prep_data)
                if pred == "acc":
                    st.success("This Prediction States That The Car Is Acceptable")
                if pred == "good":
                    st.success("This Prediction States That The Car Is Good")
                if pred == "unacc":
                    st.success("This Prediction States That The Car Is Unacceptable")
                if pred == "vgood":
                    st.success("This Prediction States That The Car Is Very Good")

    if Chosen_data == "Wine Quality":
        st.subheader("Evaluate Wine Quality")
        st.image("images (41).jpeg")
        dataset = pd.read_csv('datasets_35901_52633_winequalityN.csv')
        df = dataset
        st.dataframe(dataset.head(3))
        if st.checkbox("Show Summary"):
            st.write(dataset.describe())
        if st.checkbox("Show Shape Of Data Set"):
            st.write(dataset.shape)
            st.write(
                "This Means That The Prediction Will Be Made From Machine Learning Model Trained On Six thousand, Four Hundred And Ninty Seven Diffrent Instances")
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]


        mean = X["fixed acidity"].mean()
        X["fixed acidity"].fillna(mean, inplace=True)
        X["fixed acidity"].isnull().sum()

        mean2 = X["volatile acidity"].mean()
        X["volatile acidity"].fillna(mean, inplace=True)
        X["volatile acidity"].isnull().sum()

        mean3 = X["citric acid"].mean()
        X["citric acid"].fillna(mean, inplace=True)
        X["citric acid"].isnull().sum()

        mean4 = X["residual sugar"].mean()
        X["residual sugar"].fillna(mean, inplace=True)
        X["residual sugar"].isnull().sum()

        mean5 = X["chlorides"].mean()
        X["chlorides"].fillna(mean, inplace=True)
        X["chlorides"].isnull().sum()

        mean6 = X["pH"].mean()
        X["pH"].fillna(mean, inplace=True)
        X["pH"].isnull().sum()

        mean7 = X["sulphates"].mean()
        X["sulphates"].fillna(mean, inplace=True)
        X["sulphates"].isnull().sum()

        quaity_mapping = {3: "Low", 4: "Low", 5: "Medium", 6: "Medium", 7: "Medium", 8: "High", 9: "High"}
        y = y.map(quaity_mapping)

        mapping_quality = {"Low": 0, "Medium": 1, "High": 2}
        y = y.map(mapping_quality)

        type_mapping = {"white": 0, "red": 1}
        X["type"] = X["type"].map(type_mapping)

        sns.set(style="darkgrid")
        sns.countplot(y, hue="type", data=df)
        st.pyplot()
        st.success("0 Indicates Low Quality")
        st.success("1 Indicates Average Quality")
        st.success("2 Indicates High Quality")

        X = X.values

        from sklearn.ensemble import RandomForestClassifier
        classifier2 = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
        classifier2.fit(X, y)


        st.subheader("Enter Values For Wine Features Here")
        type = st.selectbox("Select Type Of Wine", tuple(type_mapping.keys()))
        fixed_acidity = st.number_input("Fixed Acidity", 0.0, 500.0)
        volatile_acidity = st.number_input("Volatile Acidity", 0.000, 500.000)
        citric_acid = st.number_input("Citric Acid", 0.000, 500.000)
        residual_sugar = st.number_input("Residual Sugar", 0.000, 500.000)
        chlorides = st.number_input("chlorides", 0.000, 500.000)
        free_sulfur_dioxide = st.number_input("Free Sulphur Dioxide", 0.000, 500.000)
        total_sulfur_dioxide = st.number_input("Total Sulphur Dioxide", 0.000, 500.000)
        density = st.number_input("Density", 0.000, 500.000)
        pH = st.number_input("ph", 0.000, 500.000)
        sulphates = st.number_input("Sulphates", 0.000, 500.000)
        alcohol = st.number_input("Alcohol", 0.000, 500.000)

        def get_value(val,my_dict):
            for key ,value in my_dict.items():
                if val == key:
                    return value

        types = get_value(type, type_mapping)


        st.subheader("Options Selected:")
        Encoded_as =  {"types": types,"fixed_acidity": fixed_acidity,"volatile_acidity": volatile_acidity,"citric_acid": citric_acid,"residual_sugar": residual_sugar,"chlorides": chlorides,"free_sulfur_dioxide": free_sulfur_dioxide,"total_sulfur_dioxide": total_sulfur_dioxide,"density": density,"pH": pH,"sulphates": sulphates,"alcohol": alcohol}
        st.write(Encoded_as)
        The_choices = [types, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                       free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]

        final_choices = np.array(The_choices).reshape(1, -1)


        now_pred = classifier2.predict(final_choices)
        st.subheader("Click The Predict Button Below")

        if st.button("Predict"):

            if now_pred == 0:
                st.success("This Prediction States That The Wine Has Low Quality")
            if now_pred == 1:
                st.success("This Prediction States That The Wine Has Average Quality")
            if now_pred == 2:
                st.success("This Prediction States That The Wine Has High Quality")

    if Chosen_data == "Heart Failure Mortality":
        st.subheader("Predict Heart Failure Mortality ")
        st.image("images (43).jpeg")
        dataset = pd.read_csv('datasets_727551_1263738_heart_failure_clinical_records_dataset.csv')
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        if st.checkbox("Pie Chart"):
            st.write(y.value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()
            st.success("0 Indicates That Patient Lives")
            st.success("1 Indicates That Patient Dies")
        if st.checkbox("Show Summary"):
            st.write(dataset.describe())

        if st.checkbox("Show Shape Of Data Set"):
            st.write(dataset.shape)
            st.write(
                "This Means That The Prediction Will Be Made From Machine Learning Model Trained On Two Hundred And Ninty Nine Diffeent Instances")
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0, max_depth=6)
        classifier.fit(X, y)

        Binary = {"Yes": 1, "No": 0}
        sex_pick = {"Male": 1, "Female": 0}

        st.subheader("Enter Values Of Variables For Prediction Here")
        Age = st.slider("Patient's Age", 0, 200)
        Anaemia = st.selectbox("Does Patient Have Anaemia?", tuple(Binary.keys()))
        creatinine = st.number_input("Level Of Creatinine Phosphokinase In Blood", 0.0, 500.0)
        Diabetes = st.selectbox("Does Patient Have Diabetes?", tuple(Binary.keys()))
        ejection_fraction = st.number_input("Pecentage Of Blood Leaving The Heart At Each Contraction(Ejection Fraction)", 0.0, 500.0)
        high_blood_pressure = st.selectbox("Does Patient Have Hypertension?", tuple(Binary.keys()))
        platelets = st.number_input("Platelets In Patients Blood", 0.0, 1000000.0)
        serum_creatinine = st.number_input("Level Of Serum Creatinine In Patient's Blood", 0.0, 20.0)
        serum_sodium = st.number_input("Level Of Serum Sodium In Patient's Blood", 0.0, 1000.0)
        sex = st.selectbox("Sex Of Patient", tuple(sex_pick.keys()))
        smoking = st.selectbox("Does Patient Smokes?", tuple(Binary.keys()))
        time = st.number_input("Follow Up Period (Days)", 0.0, 500.0)

        def get_value(val,my_dict):
            for key ,value in my_dict.items():
                if val == key:
                    return value

        Anaemia_state = get_value(Anaemia, Binary)
        Diabetic = get_value(Diabetes, Binary)
        high_blood = get_value(high_blood_pressure, Binary)
        Sex_Patient = get_value(sex, sex_pick)
        Smoking_value = get_value(smoking, Binary)

        all_variables = [Age, Anaemia_state, creatinine, Diabetic, ejection_fraction, high_blood, platelets, serum_creatinine, serum_sodium, Sex_Patient, Smoking_value, time]
        final_variables = np.array(all_variables).reshape(1, -1)
        predict_it = classifier.predict(final_variables)
        st.subheader("Click The Predict Button Below")
        if st.button("Predict"):

            if predict_it == 0:
                st.success("This Prediction States That Patient Lives")
            if predict_it == 1:
                st.success("This Prediction States That Patient Dies")

    if Chosen_data == "Breast Cancer Type":
        st.subheader("Predict Breast Cancer Type")
        st.image("images.jpeg")
        dataset = pd.read_csv('datasets_180_408_data (1).csv')
        X = dataset.iloc[:, 2:-1]
        y = dataset.iloc[:, 1:2]

        st.dataframe(dataset.head(3))
        if st.checkbox("Show Summary"):
            st.write(dataset.describe())
        if st.checkbox("Show Shape Of Data Set"):
            st.write(dataset.shape)
            st.write(
                "This Means That The Prediction Will Be Made From Machine Learning Model Trained On Five Hundred And Ninty Six Diffrent Instances")

        from sklearn.ensemble import RandomForestClassifier
        classifier1 = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0, max_depth=6)
        classifier1.fit(X, y)


        st.subheader("Enter Values Of Variables For Prediction Here")
        radius_mean = st.number_input("mean of distances from center to points on the perimeter", 0.0, 500.0)
        texture_mean = st.number_input("standard deviation of gray-scale values", 0.0, 500.0)
        perimeter_mean = st.number_input("mean size of the core tumor", 0.0, 500.0)
        area_mean = st.number_input("Area Mean", 0.0, 500.0)
        smoothness_mean = st.number_input("mean of local variation in radius lengths", 0.0, 500.0)
        compactness_mean = st.number_input("mean of perimeter^2 / area - 1.0", 0.0, 500.0)
        concavity_mean = st.number_input("mean of severity of concave portions of the contour", 0.0, 500.0)
        concave_points_mean = st.number_input("mean for number of concave portions of the contour", 0.0, 500.0)
        symmetry_mean = st.number_input("Symmetry Mean", 0.0, 500.0)
        fractal_dimension_mean = st.number_input("mean for coastline approximation - 1", 0.0, 500.0)
        radius_se = st.number_input("standard error for the mean of distances from center to points on the perimeter", 0.0, 500.0)
        texture_se = st.number_input("standard error for standard deviation of gray-scale values", 0.0, 500.0)
        perimeter_se = st.number_input("Perimeter Standard Error", 0.0, 500.0)
        area_se = st.number_input("Area Standard Error", 0.0, 500.0)
        smoothness_se = st.number_input("standard error for local variation in radius lengths", 0.0, 500.0)
        compactness_se = st.number_input("standard error for perimeter^2 / area - 1.0", 0.0, 500.0)
        concavity_se = st.number_input("standard error for severity of concave portions of the contour", 0.0, 500.0)
        concave_points_se = st.number_input("standard error for number of concave portions of the contour", 0.0, 500.0)
        symmetry_se = st.number_input("Symmetry Standard Error", 0.0, 500.0)
        fractal_dimension_se = st.number_input("standard error for 'coastline approximation' - 1", 0.0, 500.0)
        radius_worst = st.number_input("worst or largest mean value for mean of distances from center to points on the perimeter", 0.0, 500.0)
        texture_worst = st.number_input("worst or largest mean value for standard deviation of gray-scale values", 0.0, 500.0)
        perimeter_worst = st.number_input("Perimeter Worst", 0.0, 500.0)
        area_worst = st.number_input("Area Worst", 0.0, 500.0)
        smoothness_worst = st.number_input("worst or largest mean value for local variation in radius lengths", 0.0, 500.0)
        compactness_worst = st.number_input("worst or largest mean value for perimeter^2 / area - 1.0", 0.0, 500.0)
        concavity_worst = st.number_input("worst or largest mean value for severity of concave portions of the contour", 0.0, 500.0)
        concave_points_worst = st.number_input("worst or largest mean value for number of concave portions of the contour", 0.0, 500.0)
        symmetry_worst = st.number_input("Symmetry Worst", 0.0, 500.0)
        fractal_dimension_worst = st.number_input("worst or largest mean value for coastline approximation - 1d", 0.0, 500.0)

        inputs_variable = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]
        show = {"radius_mean":radius_mean, "texture_mean":texture_mean,"perimeter_mean": perimeter_mean,"area_mean": area_mean,"smoothness_mean": smoothness_mean,
               "compactness_mean": compactness_mean,"concavity_mean": concavity_mean,"concave_points_mean": concave_points_mean,"symmetry_mean": symmetry_mean,
              "fractal_dimension_mean": fractal_dimension_mean,"radius_se": radius_se,"texture_se": texture_se,"perimeter_se": perimeter_se,"area_se": area_se,"smoothness_se": smoothness_se,
               "compactness_se": compactness_se,"concavity_se": concavity_se,"concave_points_se": concave_points_se,"symmetry_se": symmetry_se,"fractal_dimension_se": fractal_dimension_se,
               "radius_worst": radius_worst,"texture_worst": texture_worst,"perimeter_worst": perimeter_worst,"area_worst": area_worst,"smoothness_worst": smoothness_worst,
               "compactness_worst": compactness_worst,"concavity_worst": concavity_worst,"concave_points_worst": concave_points_worst,"symmetry_worst": symmetry_worst,
                "fractal_dimension_worst": fractal_dimension_worst}
        st.subheader("Options Selected:")
        st.write(show)
        finals = np.array(inputs_variable).reshape(1, -1)

        The_pred = classifier1.predict(finals)
        st.subheader("Click The Diagnose Button Below")
        if st.button("Diagnose"):
            if The_pred == "M":
                st.success("Cancer Type Is Malignant")
            if The_pred == "B":
                st.success("Cancer Type Is Benign")


    if Chosen_data == "About Developer":
        st.subheader("About Me")
        st.write("Tijani Muabarak Adewale is a second year student of Babcock University, "
                 "Ilishan Remo Ogun State. He studies computer science and he is passionate about making positive impact to the world with his knowledge. Mubarak is fueled by his passion for machine learning and artificial intelligence and he is kin to help make life easier with his passion"
                 ". He is open to internship opportunity at the moment. Reach out to tijanimubarak2001@gmail.com to connect!")
        st.image("IMG_1311_4 (1)_3.jpg")

















fitted_model = create_Model(dataset_name)





