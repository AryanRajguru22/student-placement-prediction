"""Plain-language copy for the app (no logic, no model access).

The dataset card (Hugging Face Krooz/Campus_Recruitment_CSV) does not define the fields precisely, so the descriptions below say what each field
generally means and state the range actually observed in the development data; they never add a meaning the dataset does not support.
"""
from __future__ import annotations

# short, student-friendly explanation shown under/next to each input (the supported range is appended by the app)
FIELD_HELP = {
    "CGPA": "Your cumulative grade point average on a 10-point scale.",
    "SSC_Marks": "Your Secondary School Certificate (class 10) result, as a percentage.",
    "HSC_Marks": "Your Higher Secondary Certificate (class 12) result, as a percentage.",
    "Internships": "How many internships you have completed. Counts are treated as categories, not as 'more is always better'.",
    "Projects": "How many academic or personal projects you have completed.",
    "Workshops/Certifications": "How many workshops or certifications you have completed.",
    "AptitudeTestScore": "Your score on an aptitude test (quantitative, logical, verbal reasoning).",
    "SoftSkillsRating": "A rating of communication and interpersonal skills, as recorded in the dataset (observed range 3.0 to 4.8).",
    "ExtracurricularActivities": "Whether you take part in extracurricular activities (clubs, sports, societies).",
    "PlacementTraining": "Whether you have attended a placement-training programme.",
}

GROUPS = {
    "Academic": ("Your marks so far.", ["CGPA", "SSC_Marks", "HSC_Marks"]),
    "Technical": ("Hands-on exposure and technical aptitude.", ["Internships", "Projects", "Workshops/Certifications", "AptitudeTestScore"]),
    "Employability": ("Soft skills and activities beyond the classroom.", ["SoftSkillsRating", "ExtracurricularActivities", "PlacementTraining"]),
}

INTRO_STEPS = [
    ("1  Enter your profile", "Ten characteristics in three groups: academic, technical and employability. Nothing is assumed: every field is required."),
    ("2  Get an estimate", "A statistical model returns a probability of being placed, and what that means at two cut-offs (0.35 and 0.50)."),
    ("3  See why", "Every factor's contribution to the estimate is shown, computed from the model's own coefficients."),
]

HOW_TO_READ = (
    "The probability is what a model trained on past records of **students with similar characteristics** would estimate. "
    "It is a pattern in one dataset, not a prediction of your own future and not a recruitment decision."
)

LIMITATIONS_SHORT = (
    "One observational dataset whose structure looks rule-based or constructed (its origin is not established), so strong results here do not show the same accuracy on real "
    "placement populations. Contributions are associations, not causes. About 1 in 5 held-out students was misclassified."
)

PRIVACY_NOTE = ("Your entries are used only to compute the result in memory while this page is open. The app does not store them, log them, "
                "use accounts or a database, or send them to any other service. When the app is hosted online, they are processed by the server that hosts it.")
