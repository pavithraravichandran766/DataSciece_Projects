import pickle
with open('model.pkl','rb') as f:
    model=pickle.load(f)
model.predict([[230.1,37.8,69.2]])