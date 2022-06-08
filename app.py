from flask import Flask, request, render_template, jsonify, make_response
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

# Main function here
# ------------------
@app.route('/')
def display():
    return render_template("prediction.html")

@app.route('/dashboard')
def show():
    return render_template("dashboard.html")


@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        clf = joblib.load("svmclf.pkl")
        
        # Get values through input bars
        accountlength = int(request.form.get("accountlength"))

        location = request.form.get("location")
        print(location)
        if (location == '445'):
            location = 1
        elif (location == '452'):
            location = 2
        elif (location == '547'):
            location = 3

        intplan = request.form.get("intplan")
        if (intplan == 'yes'):
            intplan = 1
        elif (intplan == 'no'):
            intplan = 0
            
        voiceplan = request.form.get("voiceplan")
        if voiceplan == 'yes':
            voiceplan = 1
        elif voiceplan == 'no':
            voiceplan = 0
        
        voicemsgs = int(request.form.get("voicemsgs"))
        servicecalls = int(request.form.get("servicecalls"))
        daymin = float(request.form.get("daymin"))
        daycalls = int(request.form.get("daycalls"))
        evemin = float(request.form.get("evemin"))
        evecalls = int(request.form.get("evecalls"))
        nightmin = float(request.form.get("nightmin"))
        nightcalls = int(request.form.get("nightcalls"))
        intmin = float(request.form.get("intmin"))
        intcalls = int(request.form.get("intcalls"))

        daycharge = daymin*0.17
        evecharge = evemin*0.085
        nightcharge = nightmin*0.045
        intcharge = intmin*0.27

        print(type(accountlength))
        print(type(location))
        print(type(intplan))
        print(type(voiceplan))
        print(type(voicemsgs))
        print(type(daymin))
        print(type(daycalls))
        print(type(servicecalls))

        
        # Put inputs to dataframe
        X = pd.DataFrame([[accountlength, location,intplan,voiceplan,voicemsgs,
        daymin,daycalls,daycharge,
        evemin,evecalls,evecharge,
        nightmin,nightcalls,nightcharge,
        intmin,intcalls,intcharge,servicecalls]],
         columns = ["account_length", "location_code","intertiol_plan","voice_mail_plan","number_vm_messages",
         "total_day_min","total_day_calls","total_day_charge",
         "total_eve_min","total_eve_calls","total_eve_charge",
         "total_night_minutes","total_night_calls","total_night_charge",
         "total_intl_minutes","total_intl_calls","total_intl_charge","customer_service_calls"])
        
        # Get prediction
        prediction = clf.predict(X)[0]
        # prediction = 1
        
    else:
        prediction = ""
    
    return render_template("output.html", output = prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)


# @app.route('/prediction')
# def display():
#     return render_template("prediction.html")


# @app.route("/prediction/submitFunction",methods=['POST'])
# def create_entry():
#     req = request.get_json()
#     print(req)
#     res = make_response(jsonify({"message": 1}), 200)
#     return res

# @app.route("/",methods=['POST'])
# def create_entry2():
#     req = request.get_json()
#     print(req)
#     res = make_response(jsonify({"message": 1}), 200)
#     return res