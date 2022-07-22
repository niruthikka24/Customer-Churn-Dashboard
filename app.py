from flask import Flask, request, render_template, jsonify, make_response
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

# Main function here
# ------------------
@app.route('/')
def display():
    return render_template("prediction2.html")

@app.route('/dashboard')
def show():
    return render_template("dashboard.html")


@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        clf = joblib.load("xgbmodel3.pkl")
        
        # Get values through input bars
        accountlength = int(request.form.get("accountlength"))

        location = request.form.get("location")
        print(location)
        # # if (location == '445'):
        # #     location = 1
        # # elif (location == '452'):
        # #     location = 2
        # # elif (location == '547'):
        # #     location = 3

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
        daycharge = float(request.form.get("daycharge"))
        evemin = float(request.form.get("evemin"))
        evecalls = int(request.form.get("evecalls"))
        evecharge = float(request.form.get("evecharge"))
        nightmin = float(request.form.get("nightmin"))
        nightcalls = int(request.form.get("nightcalls"))
        nightcharge = float(request.form.get("nightcharge"))
        intmin = float(request.form.get("intmin"))
        intcalls = int(request.form.get("intcalls"))
        intcharge = float(request.form.get("intcharge"))

        # accountlength= 91
        # servicecalls = 2
        # daymin = 258.4
        # daycalls = 112
        # daycharge = 42.89
        # evemin = 190
        # evecalls = 93
        # evecharge = 14.47
        # nightmin = 158.6
        # nightcalls = 105
        # nightcharge = 7.09
        # intmin = 12.1
        # intcalls = 3
        # intcharge = 3.27
        # intplan = 1
        # voiceplan = 0
        # voicemsgs = 0

        # daycharge = daymin*0.17
        # evecharge = evemin*0.085
        # nightcharge = nightmin*0.045
        # intcharge = intmin*0.27

        print(accountlength)
        print(location)
        print(intplan)
        print(voiceplan)
        print(voicemsgs)
        print(daymin)
        print(daycalls)
        print(servicecalls)

        totalcharge = daycharge + evecharge + nightcharge
        totalcalls = daycalls + evecalls + nightcalls
        totalmin = daymin + evemin + nightmin

        # Put inputs to dataframe
        X = pd.DataFrame([[accountlength, intplan,voiceplan,voicemsgs,
        intmin,intcalls,intcharge,servicecalls,totalcharge,totalcalls,totalmin]],
         columns = ["account_length", "intertiol_plan","voice_mail_plan","number_vm_messages",
         "total_intl_minutes","total_intl_calls","total_intl_charge","customer_service_calls",
         "total_charge","total_calls","total_min"])
        
        # Get prediction
        prediction = clf.predict(X)[0]
        print(prediction)
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