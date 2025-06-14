from youtube_vector import create_vector_from_yt_url
from qa_engine import get_response_from_query

if __name__ == "__main__":
    video_url = ""  # Insert YouTube video URL
    query = ""      # Insert user query

    if not video_url or not query:
        print("Please provide a YouTube video URL and a query.")
    else:
        db = create_vector_from_yt_url(video_url)
        response, docs = get_response_from_query(db, query)
        print(response)
