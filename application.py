from flask import Flask,render_template,request,redirect
from flask import jsonify
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('model.pkl','rb'))
car=pd.read_csv('cars_cleaned.csv')


@app.route('/',methods=['GET','POST'])
def index():
    companies=sorted(car['manufacturer'].unique())
    car_models=sorted(car['model'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel'].unique()
    conditions=car['condition'].unique()
    transmissions=car['transmission'].unique()
    cylinders=car['cylinders'].unique()
    paint_colors=car['paint_color'].unique()
    drives=car['drive'].unique()
    title_status=car['title_status'].unique()
    sizes=car['size'].unique()
    types=car['type'].unique()
    

    companies.insert(0,'Select Company')
    return render_template('index.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type,conditions=conditions,transmissions=transmissions,cylinders=cylinders,paint_colors=paint_colors,drives=drives,title_status=title_status,sizes=sizes,types=types)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
   
    company=request.form.get('company')

    car_model=request.form.get('car_models')
    year=int(request.form.get('year'))
    fuel_type=request.form.get('fuel_types')
    km_driven=request.form.get('kilo_driven')
    condition=request.form.get('conditions')
    transmission=request.form.get('transmissions')
    cylinder=request.form.get('cylinders')
    paint_color=request.form.get('paint_color')
    drive=request.form.get('drives')
    title_status=request.form.get('title_status')
    size=request.form.get('sizes')
    types=request.form.get('types')

    prediction=model.predict(pd.DataFrame([[car_model,company,year,km_driven,fuel_type,condition,transmission,cylinder,paint_color,drive,title_status,size,types]], columns=[ 'model','manufacturer','year','odometer','fuel','condition','transmission','cylinders','paint_color','drive','type','size','title_status']))
                              
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__=='__main__':
    app.run(debug=True)