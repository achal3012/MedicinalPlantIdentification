from flask import Flask, render_template, request, redirect, url_for
import mysql.connector
import os

from dotenv import load_dotenv

load_dotenv()

import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

model = tf.keras.models.load_model(
    "model/medicinal_plant_mobilenetv2.keras"
)

class_names = [
    "Aloevera",
    "Amla",
    "Ashwagandha",
    "Ginger",
    "Mint",
    "Moringa",
    "Neem",
    "Tulsi",
    "Turmeric"
]

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_image():

    if 'plant_image' not in request.files:
        return redirect(url_for('home'))

    file = request.files['plant_image']

    if file.filename == '':
        return redirect(url_for('home'))

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(filepath)

    # Image preprocessing
    img = load_img(
        filepath,
        target_size=(224, 224)
    )

    img_array = img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(img_array)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    print("Predicted plant:", predicted_class)

    # Get plant information from MySQL
    db_connection = get_db_connection()
    cursor = db_connection.cursor(dictionary=True)

    query = """
    SELECT plant_name, scientific_name, description, uses, image, health_problem
    FROM plants
    WHERE LOWER(plant_name) = LOWER(%s)
    """

    cursor.execute(
        query,
        (predicted_class,)
    )

    result = cursor.fetchone()

    print("Database result:", result)

    cursor.close()

    return render_template(
        'result.html',
        result=result,
        prediction=predicted_class
    )

print("DB_HOST:", os.getenv("DB_HOST"))
print("DB_PORT:", os.getenv("DB_PORT"))
print("DB_USER:", os.getenv("DB_USER"))
print("DB_NAME:", os.getenv("DB_NAME"))

db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", "26114")),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
    ssl_disabled=False
)

def get_db_connection():
    global db

    try:
        if not db.is_connected():
            db.reconnect(attempts=3, delay=2)
    except mysql.connector.Error:
       db = mysql.connector.connect(
          host=os.getenv("DB_HOST"),
          port=int(os.getenv("DB_PORT", "26114")),
          user=os.getenv("DB_USER"),
          password=os.getenv("DB_PASSWORD"),
          database=os.getenv("DB_NAME"),
          ssl_disabled=False
       )

    return db

@app.route('/', methods=['GET', 'POST'])
def home():

    if request.method == 'POST':

        plant = request.form.get(
            'plant_name',
            ''
        ).strip()

        health_problem = request.form.get(
            'health_problem',
            ''
        ).strip()

        db_connection = get_db_connection()
        cursor = db_connection.cursor(dictionary=True)

        if health_problem:

            query = """
            SELECT plant_name, scientific_name, description, uses, image, health_problem
            FROM plants
            WHERE LOWER(health_problem) LIKE LOWER(%s)
            """

            cursor.execute(
                query,
                (f"%{health_problem}%",)
            )

            result = cursor.fetchone()

            cursor.close()

            return render_template(
                'result.html',
                result=result
            )

        elif plant:

            query = """
            SELECT plant_name, scientific_name, description, uses, image, health_problem
            FROM plants
            WHERE LOWER(plant_name) = LOWER(%s)
            """

            cursor.execute(
                query,
                (plant,)
            )

            result = cursor.fetchone()

            cursor.close()

            return render_template(
                'result.html',
                result=result
            )

        cursor.close()

    return render_template('index.html')


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')


if __name__ == '__main__':
    app.run(debug=True)