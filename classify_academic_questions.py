import dotenv
import dspy


dotenv.load_dotenv(".env")

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class ClassifyDiscussionForum(dspy.Signature):
    """Classify the discussion forum post into one of the following categories based on the content of the forum post and guidelines from the instructor for the post (if applicable):
1. Academic Question - Questions about academic content posed to the teaching staff;
2. Academic Discussion - Usually a discussion forum post required by an assignment, or any type of discussion that does not contain an obvious question;
3. Logistics Question - Questions about course logistics;
4. Logistics Discussion - Other types of posts about course logistics;
5. Social - Discussion forum posts for social purposes."""
    post_content = dspy.InputField(desc="The content of the student's discussion forum post.")
    post_topic_content = dspy.InputField(desc="The guidelines from the instructor for this post.")
    post_classification = dspy.OutputField(desc="The classification for the post.")


if __name__ == "__main__":
    post_content = "[content of the post]"
    post_topic = "[instructor guidelines for this specific discussionn topic]"
    
    classifier = dspy.ChainOfThought(ClassifyDiscussionForum)
    classification = classifier(post_content=post_content, post_topic_content=post_topic).post_classification