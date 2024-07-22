<div align="center">
  <h1>VolleyBall Referee Hand Signals Recognition </h1>
</div>

# Introduction 🏐

<div align="center">
  <img src="examples/First.gif" alt="Example" width="700"/>
</div>

# Hand Signals 
<div align="center">
  <img src="assets/actions.png" alt="Interface" width="1000"/>
</div>

# Installation ✨

Use a dedicated environnement to install the librairies.

Clone repo :
```
git clone https://github.com/MaxLSB/Volleyball-Referee-Action-Recognition-Model.git
```
Install the requirements for the app:
```
pip install -r requirements.txt
```
# Test 

Detection with your webcam in real time:
```
python main.py
```
_(You can adjust ViewProbabilities and ViewLandmarks' values at the bottom of the file. They are set to 'True' by Default)_

# Streamlit local server 

<div align="center">
  <img src="assets/streamlit-1.png" alt="Example" width="800" />
</div>

Download the 2 example videos:
```
bash downloadExamples.sh
```
Lauching the streamlit local server (download the 2 example videos for a better experience):
```
streamlit run app.py
```

# Train the model with new actions 

Change the actions in the 'utils/config.py' file:

Start the Data Collecting process with your webcam (takes some time):
```
python data.py
```
Train the model:
```
python train.py
```

_(Scripts must be executed from the root folder of the project, be careful about the paths!)_
