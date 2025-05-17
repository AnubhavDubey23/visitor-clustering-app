from flask import Flask
import os

def create_app():
    app = Flask(__name__,static_folder='../static')
    app.secret_key = 'your-secret-key'

    # Ensure the outputs directory exists
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    from .routes import main
    app.register_blueprint(main)

    return app
