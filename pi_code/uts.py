default_system = 'You are an experienced driver who can answer questions based on perceptual images.'

def process_comment_question(question):
    if question == "What's your comment on this scene?":
        ques = []
        ques.append("Please describe the current scene.")
        ques.append("What are the important objects in the current scene, determine their status, and predict their future status.")
        ques.append("What object should the ego vehicle notice?")
        ques.append("What is the priority of the objects that the ego vehicle should consider?")
        ques.append("Are there any safety issues in the current scene?")
        ques.append("What are the safe actions to take for the ego vehicle?")
        ques.append("What are the dangerous actions to take for the ego vehicle?")
        ques.append("Predict the behavior of the ego vehicle.")
        n_q = " ".join(ques)
        return n_q
    else:
        return question