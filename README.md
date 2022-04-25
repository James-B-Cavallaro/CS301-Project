# The problem that we are investigating is as follows: testing medical students on how well they write patient notes, which is the description of the patients symptoms, is inefficient. It requires too much time and resources to grade these exams. Thus our objective is to identify concepts, given from an exam rubric, within the patient notes in order to autonomously grade exams. This is an interesting concept as it would improve the grading process; firstly, it would mitigate the potential for error and bias from the grader, as there wouldn't be any way for them to interpret the notes incorrectly. Additionally, by automatically mapping clinical concepts to notes, it will make the development of these assignments easier in the future, as well as making them easier to grade in general. The data that will be used is supplied by the NBME, the National Board of Medical Examiners. This data is taken from the USMLE Step 2 examination, which tests students on their ability to write patient notes. Our current plan is to use a random forest as the primary learning method, training the algorithm with samples of notes and how they match to different clinical concepts. To test the performance of the algorithm, we plan on taking the annotated patient notes and using them for preliminary tests, comparing the algorithm's accuracy in mapping notes to features to the annotations. And if there is an existing algorithm we will use it, because it will greatly reduce our time. The evaluation of the results requires the provision of a large amount of data, and when the data displayed after the success of the graph can satisfy the provided data, it is successful.
Competition Link: https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/overview
