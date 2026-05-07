from flask import Flask, render_template, request
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy
from transformers import pipeline
import os

app = Flask(__name__)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///resume.db'

db = SQLAlchemy(app)

# Upload Folder
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# AI Feedback Model
generator = pipeline(
    "text-generation",
    model="gpt2"
)

# Skill List
skills = [
    "python",
    "java",
    "sql",
    "machine learning",
    "html",
    "css",
    "javascript",
    "react",
    "flask"
]

# Database Table
class Resume(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    score = db.Column(db.Float)

    skills = db.Column(db.String(500))

    missing = db.Column(db.String(500))


# Home Page
@app.route('/')
def home():

    return render_template('index.html')


# Analyze Resume
@app.route('/analyze', methods=['POST'])
def analyze():

    # Get uploaded resume
    resume_file = request.files['resume']

    # Get job description
    job_description = request.form['job']

    # Save uploaded file
    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        resume_file.filename
    )

    resume_file.save(filepath)

    # Extract text from PDF
    text = ""

    with pdfplumber.open(filepath) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    # Convert to lowercase
    resume_text = text.lower()

    job_text = job_description.lower()

    # TF-IDF Similarity
    documents = [resume_text, job_text]

    tfidf = TfidfVectorizer()

    matrix = tfidf.fit_transform(documents)

    score = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )

    match_percentage = round(
        score[0][0] * 100,
        2
    )

    # Find detected skills
    found_skills = []

    for skill in skills:

        if skill in resume_text:
            found_skills.append(skill)

    # Find missing skills
    missing_skills = []

    for skill in skills:

        if skill in job_text and skill not in found_skills:
            missing_skills.append(skill)

    # Basic Feedback
    feedback = []

    if match_percentage > 80:

        feedback.append(
            "Excellent Resume Match"
        )

    elif match_percentage > 60:

        feedback.append(
            "Good Resume Match"
        )

    else:

        feedback.append(
            "Needs Improvement"
        )

    if missing_skills:

        feedback.append(
            "Add missing skills"
        )

    # AI Feedback Generation
    prompt = f"""
    Give professional resume improvement suggestions
    for this candidate based on missing skills:
    {missing_skills}
    """

    ai_response = generator(
        prompt,
        max_length=80,
        num_return_sequences=1
    )

    ai_feedback = ai_response[0]['generated_text']

    # Save to Database
    new_resume = Resume(
        score=match_percentage,
        skills=str(found_skills),
        missing=str(missing_skills)
    )

    db.session.add(new_resume)

    db.session.commit()

    # Return Result Page
    return render_template(
        'result.html',
        score=match_percentage,
        skills=found_skills,
        missing=missing_skills,
        feedback=feedback,
        ai_feedback=ai_feedback
    )


# Create Database
with app.app_context():

    db.create_all()


# Run App
if __name__ == '__main__':

    app.run(debug=True)