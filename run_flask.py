from flask import Flask, render_template, redirect, request, url_for
from test import Test
app = Flask(__name__)

@app.route('/')
@app.route('/<prime>')
def jongsuldong(prime=None):
	return render_template('jongsuldong.html', prime=prime)

@app.route('/project', methods=['POST'])
def project_page(prime=None):
	if request.method == 'POST':
		starting_word = request.form['prime']
		theme = int(request.form['chk_info'])
		print("starting word : " + starting_word)
		print("method POST executed")
		# if not myString:
		if not starting_word:
			print("no starting word")
			poem_generated = None
		else:
			print("starting word is not none")
			poem = Test(prime = starting_word, atom = theme, length = 400)
			poem_generated = poem.run()
	else:
		print("not POST method")
		poem_generated = None
	return redirect(url_for('jongsuldong', prime=poem_generated))

if __name__ == '__main__':
	app.run(host = '0.0.0.0', port=6688, debug = True)
