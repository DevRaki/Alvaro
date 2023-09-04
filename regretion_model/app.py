from flask import Flask, render_template, request
import joblib


model = joblib.load('modelos/modelo_petalos.pkl') 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    w_sepal = float(request.form['W_sepal'])
    l_sepal = float(request.form['L_sepal'])
    w_petal = float(request.form['W_petal'])
    l_petal = float(request.form['L_petal'])
  
    pred_probabilities = model.predict_proba([[l_sepal,w_sepal,l_petal, w_petal ]])

    class_names = model.classes_

    mensaje = ""
    for i, class_name in enumerate(class_names):
        prob = pred_probabilities[0, i] * 100
        mensaje += f"Probabilidad de {class_name}: {round(prob, 2)}% <br/>"

    return render_template('result.html', pred=mensaje)


if __name__ == '__main__':
    app.run()