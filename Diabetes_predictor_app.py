{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ee6ee844-04a8-47e3-ac1c-08cf826468c0",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'streamlit'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[2], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mstreamlit\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mst\u001b[39;00m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mpickle\u001b[39;00m\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mnumpy\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mnp\u001b[39;00m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'streamlit'"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "# Custom CSS for styling\n",
    "st.markdown(\n",
    "    \"\"\"\n",
    "    <style>\n",
    "        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');\n",
    "\n",
    "        .header-title {\n",
    "            font-family: 'Poppins', sans-serif;\n",
    "            font-size: 2rem;\n",
    "            font-weight: bold;\n",
    "            color: #ffffff;\n",
    "            display: flex;\n",
    "            align-items: center;\n",
    "            gap: 10px;\n",
    "        }\n",
    "        .header-subtitle {\n",
    "            font-family: 'Poppins', sans-serif;\n",
    "            font-size: 1rem;\n",
    "            color: #ff9900;\n",
    "        }\n",
    "        .divider {\n",
    "            border-top: 1px solid #ffffff;\n",
    "            margin: 15px 0;\n",
    "        }\n",
    "        .metric-title {\n",
    "            font-family: 'Poppins', sans-serif;\n",
    "            font-size: 1rem;\n",
    "            font-weight: bold;\n",
    "            color: #ffffff;\n",
    "            margin-bottom: 3px;\n",
    "        }\n",
    "        .metric-desc {\n",
    "            font-family: 'Poppins', sans-serif;\n",
    "            font-size: 0.8rem;\n",
    "            color: #dddddd;\n",
    "            margin-bottom: 2px;\n",
    "        }\n",
    "        .metric-range {\n",
    "            font-family: 'Poppins', sans-serif;\n",
    "            font-size: 0.8rem;\n",
    "            color: #4CAF50;\n",
    "            margin-bottom: 5px;\n",
    "        }\n",
    "    </style>\n",
    "    \"\"\",\n",
    "    unsafe_allow_html=True\n",
    ")\n",
    "\n",
    "# Header section\n",
    "st.markdown(\n",
    "    \"\"\"\n",
    "    <div class=\"header-title\">\n",
    "        Diabetes Predictor\n",
    "    </div>\n",
    "    <div class=\"header-subtitle\">\n",
    "        Assess your risk of diabetes with our smart analysis.\n",
    "    </div>\n",
    "    <div class=\"divider\"></div>\n",
    "    \"\"\",\n",
    "    unsafe_allow_html=True\n",
    ")\n",
    "\n",
    "# Load the diabetes model\n",
    "try:\n",
    "    with open(\"diabetes_model.pkl\", \"rb\") as f:\n",
    "        model = pickle.load(f)\n",
    "except Exception as e:\n",
    "    st.error(f\"Error loading model: {e}\")\n",
    "    st.stop()\n",
    "\n",
    "# Load the robust scaler\n",
    "try:\n",
    "    with open(\"robust_scaler.pkl\", \"rb\") as f:\n",
    "        scaler = pickle.load(f)\n",
    "except Exception as e:\n",
    "    st.error(f\"Error loading scaler: {e}\")\n",
    "    st.stop()\n",
    "\n",
    "# Define feature descriptions and suggested ranges (modify as needed)\n",
    "feature_info = {\n",
    "    \"Pregnancies\": {\"desc\": \"Number of times pregnant.\", \"range\": \"(0 - 17)\"},\n",
    "    \"Glucose\": {\"desc\": \"Plasma glucose concentration (mg/dL).\", \"range\": \"(80 - 200)\"},\n",
    "    \"BloodPressure\": {\"desc\": \"Diastolic blood pressure (mm Hg).\", \"range\": \"(40 - 120)\"},\n",
    "    \"SkinThickness\": {\"desc\": \"Triceps skin fold thickness (mm).\", \"range\": \"(10 - 99)\"},\n",
    "    \"Insulin\": {\"desc\": \"2-Hour serum insulin (mu U/ml).\", \"range\": \"(15 - 276)\"},\n",
    "    \"BMI\": {\"desc\": \"Body mass index (kg/m²).\", \"range\": \"(15 - 67)\"},\n",
    "    \"DiabetesPedigreeFunction\": {\"desc\": \"Diabetes pedigree function.\", \"range\": \"(0.078 - 2.42)\"},\n",
    "    \"Age\": {\"desc\": \"Age (years).\", \"range\": \"(21 - 81)\"}\n",
    "}\n",
    "\n",
    "# Create inputs for each feature\n",
    "inputs = []\n",
    "for feature, details in feature_info.items():\n",
    "    st.markdown(f\"<div class='metric-title'>{feature}</div>\", unsafe_allow_html=True)\n",
    "    st.markdown(f\"<div class='metric-desc'>{details['desc']}</div>\", unsafe_allow_html=True)\n",
    "    st.markdown(f\"<div class='metric-range'>{details['range']}</div>\", unsafe_allow_html=True)\n",
    "    value = st.number_input(\"\", min_value=0.0, format=\"%.2f\", key=feature)\n",
    "    inputs.append(value)\n",
    "\n",
    "# Convert input list to numpy array\n",
    "input_array = np.array(inputs).reshape(1, -1)\n",
    "\n",
    "# When the user clicks the prediction button\n",
    "if st.button(\"Predict Diabetes\"):\n",
    "    try:\n",
    "        # Scale the inputs using the robust scaler\n",
    "        input_scaled = scaler.transform(input_array)\n",
    "        # Make the prediction\n",
    "        prediction = model.predict(input_scaled)\n",
    "        # Display result based on the prediction outcome\n",
    "        if prediction[0] == 1:\n",
    "            st.markdown(\"🚨 **High Risk of Diabetes**: Please consult with a healthcare provider.\")\n",
    "        else:\n",
    "            st.markdown(\"🩺 **Low Risk of Diabetes**: Your results are within the normal range.\")\n",
    "    except Exception as e:\n",
    "        st.error(f\"Prediction error: {e}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7529a91d-6dcb-46eb-9dc4-cf29ec1d609d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
