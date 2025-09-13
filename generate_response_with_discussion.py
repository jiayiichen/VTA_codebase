import pandas
import dspy
import json
import os
import numpy as np
from openai import OpenAI

# semantic similarity - TODO: add better metrics (?)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tqdm

import dotenv
dotenv.load_dotenv(".env")

# Context dependency classifier
class ClassifyContextDependence(dspy.Signature):
    """You are a helpful teaching assistant. Given a discussion forum post and the guidelines from the instructor for this post, determine whether you require more context for responding to the post."""
    post_content = dspy.InputField(desc="The content of the student's discussion forum post.")
    post_topic_content = dspy.InputField(desc="The guidelines from the instructor for this post.")
    course_title = dspy.InputField(desc="The title or the course number of the course")
    course_desc = dspy.InputField(desc="The description of the course, when applicable")
    need_context = dspy.OutputField(desc="Whether you need additional context. Answer with yes or no.")

# Original TA prompt
TA_PROMPT = """You are a virtual teaching assistant for a course called COURSE_NAME. COURSE_DESCRIPTION Respond to the discussion forum post for this course provided by one of the students. Please offer the response based on your existing knowledge base. Please add a general greeting in each response.

Please adhere to the following pedagogical goals:
1. Clarify Misunderstandings: Support knowledge acquisition by articulating questions, addressing confusion, and receiving clarifications from peers or instructors.
2. Deepen Disciplinary Understanding: Promote deeper engagement with core concepts and themes through elaboration, critical questioning, and interaction with diverse perspectives.
3. Develop Higher-Order Thinking: Cultivate critical thinking and reasoning skills by analyzing ideas, justifying positions, synthesizing information, and exploring alternative viewpoints.
4. Enhance Metacognitive Awareness: Strengthen self-regulated learning by reflecting on one's understanding, identifying gaps in knowledge, and evaluating the quality of reasoning.
5. Foster Collaborative Knowledge Construction and Social Presence: Fosters peer interaction and collective learning by connecting diverse student perspectives, encouraging the exchange of ideas, and supporting collaborative knowledge construction, positioning the discussion forum as a shared space for dialogue and co-construction of understanding."""

# TA prompt with context
TA_PROMPT_WITH_CONTEXT = """You are a virtual teaching assistant for a course called COURSE_NAME. COURSE_DESCRIPTION

You are responding to the following discussion forum post for this course provided by one of the students. This post is CONTEXT_TYPE.

DISCUSSION_CONTEXT

Please offer the response based on your existing knowledge base. Please add a general greeting in each response.

Please adhere to the following pedagogical goals:
1. Clarify Misunderstandings: Support knowledge acquisition by articulating questions, addressing confusion, and receiving clarifications from peers or instructors.
2. Deepen Disciplinary Understanding: Promote deeper engagement with core concepts and themes through elaboration, critical questioning, and interaction with diverse perspectives.
3. Develop Higher-Order Thinking: Cultivate critical thinking and reasoning skills by analyzing ideas, justifying positions, synthesizing information, and exploring alternative viewpoints.
4. Enhance Metacognitive Awareness: Strengthen self-regulated learning by reflecting on one's understanding, identifying gaps in knowledge, and evaluating the quality of reasoning.
5. Foster Collaborative Knowledge Construction and Social Presence: Fosters peer interaction and collective learning by connecting diverse student perspectives, encouraging the exchange of ideas, and supporting collaborative knowledge construction, positioning the discussion forum as a shared space for dialogue and co-construction of understanding."""


# TA prompt with similar posts
TA_PROMPT_WITH_SIMILAR = """You are a virtual teaching assistant for a course called COURSE_NAME. COURSE_DESCRIPTION

You are responding to the following discussion forum post for this course provided by one of the students. This post is CONTEXT_TYPE.

DISCUSSION_CONTEXT

Additionally, here are some other relevant posts from students on this same topic:
SIMILAR_POSTS

Please offer the response based on your existing knowledge base. Please add a general greeting in each response.

When appropriate, refer to insights or perspectives from these related posts to foster connections between student ideas.

Please adhere to the following pedagogical goals:
1. Clarify Misunderstandings: Support knowledge acquisition by articulating questions, addressing confusion, and receiving clarifications from peers or instructors.
2. Deepen Disciplinary Understanding: Promote deeper engagement with core concepts and themes through elaboration, critical questioning, and interaction with diverse perspectives.
3. Develop Higher-Order Thinking: Cultivate critical thinking and reasoning skills by analyzing ideas, justifying positions, synthesizing information, and exploring alternative viewpoints.
4. Enhance Metacognitive Awareness: Strengthen self-regulated learning by reflecting on one's understanding, identifying gaps in knowledge, and evaluating the quality of reasoning.
5. Foster Collaborative Knowledge Construction and Social Presence: Fosters peer interaction and collective learning by connecting diverse student perspectives, encouraging the exchange of ideas, and supporting collaborative knowledge construction, positioning the discussion forum as a shared space for dialogue and co-construction of understanding."""

STUDENT_PROMPT = """You are a student in a course called COURSE_NAME. Respond to the discussion forum post for this course provided by a fellow student."""


# Here you need to host your own Llama-3-70B-Instruct
# You can do it through SGLang or VLLM
llama = dspy.LM(model="openai/meta-llama/Meta-Llama-3-70B-Instruct", api_key="", api_base="http://127.0.0.1:8000/v1", max_tokens=1000, cache=True)
dspy.configure(lm=llama, provide_traceback=True)

def classify_context_dependence(df):
    """
    Classify posts as context-dependent or independent.

    Args:
        df: DataFrame containing the posts to classify

    Returns:
        DataFrame with context_independence column added
    """
    
    dependence_judge = dspy.ChainOfThought(ClassifyContextDependence)

    all_class = []
    for i, row in df.iterrows():
        classification = dependence_judge(
            post_content=row["discussion_post_content_clean"],
            post_topic_content=row["discussion_topic"],
            course_title=row["course_title"],
            course_desc=row["course_desc"]
        ).need_context
        classification = int(classification.lower().startswith("no"))
        all_class.append(classification)

    df["context_independence"] = all_class
    return df

def respond_to_post(course_prompt, discussion_topic, post_content):
    """Generate a response using the LLM with the given prompt and content."""
    discussion_content = f"**Discussion Instructions:** {discussion_topic}\n**Post Content:** {post_content}"
    return llama(messages=[
                {
                    "role": "system",
                    "content": course_prompt,
                },
                {
                    "role": "user",
                    "content": discussion_content
                }
            ])[0]

def format_ta_prompt(course_name, course_description):
    """Format the basic TA prompt with course name and description."""
    course_prompt = TA_PROMPT.replace("COURSE_NAME", course_name)
    if course_description is None:
        course_prompt = course_prompt.replace("COURSE_DESCRIPTION", "")
    else:
        course_prompt = course_prompt.replace("COURSE_DESCRIPTION", f"The course has the following description: {course_description}")
    return course_prompt

def format_ta_prompt_with_context(course_name, course_description, parent_content=None):
    """Format TA prompt with context (parent post) if available."""
    course_prompt = TA_PROMPT_WITH_CONTEXT.replace("COURSE_NAME", course_name)

    # course description
    if course_description is None:
        course_prompt = course_prompt.replace("COURSE_DESCRIPTION", "")
    else:
        course_prompt = course_prompt.replace("COURSE_DESCRIPTION", f"The course has the following description: {course_description}")

    # context information
    if parent_content is not None:
        course_prompt = course_prompt.replace("CONTEXT_TYPE", "a reply to another student's post")
        course_prompt = course_prompt.replace("DISCUSSION_CONTEXT", f"Parent post: {parent_content}")
    else:
        course_prompt = course_prompt.replace("CONTEXT_TYPE", "an initial post in the discussion")
        course_prompt = course_prompt.replace("DISCUSSION_CONTEXT", "")

    return course_prompt

def format_ta_prompt_with_similar(course_name, course_description, similar_posts, parent_content=None):
    """Format TA prompt with context and similar posts."""
    course_prompt = TA_PROMPT_WITH_SIMILAR.replace("COURSE_NAME", course_name)

    # course description
    if course_description is None:
        course_prompt = course_prompt.replace("COURSE_DESCRIPTION", "")
    else:
        course_prompt = course_prompt.replace("COURSE_DESCRIPTION", f"The course has the following description: {course_description}")

    # context information
    if parent_content is not None:
        course_prompt = course_prompt.replace("CONTEXT_TYPE", "a reply to another student's post")
        course_prompt = course_prompt.replace("DISCUSSION_CONTEXT", f"Parent post: {parent_content}")
    else:
        course_prompt = course_prompt.replace("CONTEXT_TYPE", "an initial post in the discussion")
        course_prompt = course_prompt.replace("DISCUSSION_CONTEXT", "")

    # similar posts
    similar_posts_text = ""
    for i, post in enumerate(similar_posts):
        similar_posts_text += f"Similar post #{i+1}: {post}\n\n"

    course_prompt = course_prompt.replace("SIMILAR_POSTS", similar_posts_text)

    return course_prompt

import numpy as np

def find_top_ten(og_post, similar_posts):
    embedder = dspy.Embedder("openai/text-embedding-3-small", batch_size=100)
    og_embed = embedder(og_post)
    new_embed = embedder(similar_posts)
    similarity_scores = []
    for i, emb in enumerate(new_embed):
        similarity = np.dot(og_embed, emb) / (np.linalg.norm(og_embed) * np.linalg.norm(emb))
        similarity_scores.append((similarity, similar_posts[i]))
    similarity_scores.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in similarity_scores[:10]]
    


def main():
    """
    Main function to run the post classification and response generation.

    Args:
        classify_posts: Whether to run the context dependency classification
    """
    df_meta = pandas.read_parquet("discussion_post_19_24_metadata.parquet.gzip")
    df_content = pandas.read_parquet("discussion_post_19_24_content.parquet.gzip")
    df_academic = pandas.read_excel("300_revised_with_context.xlsx")
    
    df_academic = classify_context_dependence(df_academic)
    
    records = json.loads(df_academic.to_json(orient="records"))
    for i, row in tqdm.tqdm(enumerate(records)):
        if records["context_independence"] == 0:
            continue
        # Get the discussion topic id
        metadata = df_meta.loc[df_meta["discussion_post_id"] == row["discussion_post_id"]] 
        discussion_topic_id = list(metadata["discussion_topic_id"])[0]
        if not pandas.notna(discussion_topic_id):
            continue
        same_discussion_topic = df_meta.loc[df_meta["discussion_topic_id"] == discussion_topic_id]
        all_same_topics = []
        for j, srow in same_discussion_topic.iterrows():
            same_post = df_content.loc[df_content["discussion_post_id"] == srow["discussion_post_id"]]
            content_post = list(same_post["discussion_post_content_clean"])[0]
            all_same_topics.append(content_post)
        post_content = row["discussion_post_content_clean"]
        all_same_topics = find_top_ten(post_content, all_same_topics)
        # Get all posts with the same discussion topic id
        discussion_topic = row["discussion_topic"]
        course_name = row["course_title"]
        course_desc = row["course_desc"]
        
        parent_content = None

        # 1. Basic prompt (original)
        ta_pr_basic = format_ta_prompt(course_name, course_desc)
        records[i]["llama_70b_ta_response"] = respond_to_post(ta_pr_basic, discussion_topic, post_content)
        # 2. Similar posts prompt
        if len(all_same_topics) > 0:
            ta_pr_similar = format_ta_prompt_with_similar(course_name, course_desc, all_same_topics, parent_content)
            records[i]["llama_70b_ta_response_with_similar"] = respond_to_post(ta_pr_similar, discussion_topic, post_content)
        records[i]["parent_discussion_post_id"] = None

    all_responses = pandas.DataFrame.from_dict(records)
    all_responses.to_excel("300_with_context.xlsx")

if __name__ == "__main__":
    # set to True to run context dependency classification, False to use pre-classified data
    main()