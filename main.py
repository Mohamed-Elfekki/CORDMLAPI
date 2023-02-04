# 1. Library imports
#import uvicorn
#.. Uvicorn is an ASGI web server implementation for Python.Until recently Python has lacked
#   a minimal low-level server/application interface for async frameworks. The ASGI specification
#   fills this gap, and means we're now able to start building a common set of tooling usable across all async frameworks

from fastapi import FastAPI

from pydantic import BaseModel
#.. Pydantic is a Python library for data modeling/parsing that has efficient error
#   handling and a custom validation mechanism.

import numpy as np
import pickle
from typing import List



# 2. Create the app object
app = FastAPI()

pickle_in = open("Cord_model.pkl", "rb")

model = pickle.load(pickle_in)



# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Testing message for CORD API!'}


# .. Class attributes are class variables that are inherited by every object of a class
class InputData(BaseModel):
    data: List[float]

# 4. Create Http Post request to get the prediction
@app.post("/predict")
def predict(data: List[float]):
    # Use the model to make a prediction
    prediction = model.predict(np.array(data).reshape(1, -1))

    return prediction[0]



# 5. Example JSON:-
#[-27535.449219, -25971.853516, -30141.794922, -19697.228516, 3157.675537, -33066.187500, -27920.703125, -9997.800781]

#.. RESPONSE: "prediction": "When I glance over my notes and records of the Sherlock Holmescases between the years ’82 and ’90, I am faced by so many which present strange and interestingfeatures that it is no easy matter to know which to choose and which to leave."

# 5.1. Example JSON:-
# [-49246.343750,  -18282.117188,  -14447.563477 , -15476.124023,  2633.571777  ,-29541.091797 , -11439.354492,  6628.856445]
#.. FRIDAY NIGHT.


# 6. Run the API with uvicorn
# 7 Will run on http://127.0.0.1:8000

#    if __name__ == '__main__':
#      uvicorn.run(app, host='127.0.0.1', port=8000)

# 7. uvicorn main:app --reload

# 8. For Local server testing the HTTP POST request use this URL:
# .. http://localhost:8000/predict