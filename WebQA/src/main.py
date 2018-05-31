# coding:utf8
import datetime

import flask
from flask import render_template, jsonify
from simpleQA import SimpleQA
import json
import random
import logging

app = flask.Flask(__name__)
# 日志系统配置
handler = logging.FileHandler('log/app.log', encoding='UTF-8')
logging_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
handler.setFormatter(logging_format)
app.logger.addHandler(handler)

start = datetime.datetime.now()
qa = SimpleQA()
qa.setup()
end = datetime.datetime.now()
print('Time of loading knowledge base and trained models: %f' % (end - start).seconds)


@app.route('/', methods=['GET', 'POST'])
def homepage():
    if flask.request.method == 'GET':
        result = {}
        return render_template("homepage.html", result=result)
    elif flask.request.method == 'POST':  # and flask.request.form.get('query', None) == "SEARCH"
        question = flask.request.form['input']
        if str(question):
            # app.logger.info(question)  # 记录用户输入
            result = qa.get_answer(question)
            if len(result['answer']) > 10:  # top 10
                result['answer'] = result['answer'][:10]
            return render_template("homepage.html", result=result)
        else:
            return render_template("homepage.html", warning="Are you kidding me ?")


@app.route('/entity', methods=['GET', 'POST'])
def entity():
    if flask.request.method == 'GET':
        return render_template('entity.html')
    else:
        question = flask.request.form['input']
        if str(question):
            # app.logger.info(question)  # 记录用户输入
            result = qa.entity_linking(question)
            if len(result['answer']) > 10:  # top 10
                result['answer'] = result['answer'][:10]
            return render_template("entity.html", result=result)
        else:
            return render_template("entity.html", warning="Are you kidding me ?")


@app.route('/relation', methods=['GET', 'POST'])
def relation():
    if flask.request.method == 'GET':
        return render_template('relation.html')
    else:
        question = flask.request.form['input']
        if str(question):
            # app.logger.info(question)  # 记录用户输入
            result = qa.relation_detection(question)
            if len(result['answer']) > 10:  # top 10
                result['answer'] = result['answer'][:10]
            return render_template("relation.html", result=result)
        else:
            return render_template("relation.html", warning="Are you kidding me ?")


@app.route('/mention', methods=['GET', 'POST'])
def mention():
    if flask.request.method == 'GET':
        return render_template('mention.html')
    else:
        question = flask.request.form['InputTextBox']
        if str(question):
            if str(flask.request.form['input']) != "recall":
                result = qa.mention_detection(question)
                return render_template("mention.html", result=result)
            else:
                result = qa.mention_detection2(question)
                return render_template("mention.html", result=result)
        else:
            return render_template("mention.html", warning="Are you kidding me ?")


@app.route('/coming_soon', methods=['GET', 'POST'])
def coming_soon():
    if flask.request.method == 'GET':
        return render_template('coming_soon.html')
    else:
        return render_template('coming_soon.html')


@app.route('/mydict', methods=['GET', 'POST'])
def mydict():
    with open("data/data.json") as f:
        json_dict = json.load(f)
        rand = random.randint(0, len(json_dict))
        print(json_dict[rand])
    # print(rand)
    # print(json_dict[rand])
    return jsonify(json_dict[rand])


@app.route('/report', methods=['GET', 'POST'])
def report():
    question = flask.request.form['input']
    with open("data/report.log", "a") as f:
        f.write(question + "\n")
    return

if __name__ == '__main__':
    # homepage()
    app.run(host='0.0.0.0', port=4001, debug=False, processes=1)  # 114.212.190.231
