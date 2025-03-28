import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
from openai import OpenAI  # Updated import for version 1.0+
from dotenv import load_dotenv
import logging
import tempfile
import mimetypes
from werkzeug.utils import secure_filename
import sys

# For PDF processing
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
    logging.warning("PyPDF2 not installed. PDF processing will not be available.")

# For DOCX processing
try:
    import docx
except ImportError:
    docx = None
    logging.warning("python-docx not installed. DOCX processing will not be available.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (especially OpenAI API key)
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("FATAL: OPENAI_API_KEY environment variable not set.")
    # You might want to exit or raise an exception here in a real app
    # raise ValueError("OPENAI_API_KEY environment variable not set.")
else:
    # Clean the API key - remove any quotes or whitespace
    OPENAI_API_KEY = OPENAI_API_KEY.strip().strip("'").strip('"')
    # Log a masked version of the key for debugging
    masked_key = OPENAI_API_KEY[:8] + "..." + OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 12 else "***"
    logger.info(f"OpenAI API key found with format: {masked_key}")
    logger.info(f"Key starts with: {OPENAI_API_KEY[:8]}")
    logger.info(f"Key length: {len(OPENAI_API_KEY)}")

# Initialize OpenAI Client (using v1.0+ syntax)
client = None
try:
    # Direct method for client initialization
    logger.info("Attempting to initialize OpenAI client with direct method...")
    import openai
    
    # Clear any proxy settings that might be causing issues
    logger.info("Clearing any proxy settings in the environment...")
    if 'http_proxy' in os.environ:
        logger.info("Removing http_proxy from environment")
        del os.environ['http_proxy']
    if 'https_proxy' in os.environ:
        logger.info("Removing https_proxy from environment")
        del os.environ['https_proxy']
    
    # Set API key without proxies
    openai.api_key = OPENAI_API_KEY
    
    # Ensure no proxies are used in the module
    openai.proxy = None
    
    # Test if the client works with a simple API call
    logger.info("Testing openai module...")
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test message"}],
        max_tokens=5
    )
    logger.info(f"OpenAI module test successful: {response.choices[0].message.content}")
    
    # If we get here, the module works, so use it
    client = openai
    logger.info("Using openai module as client")
    
except Exception as e:
    logger.error(f"Failed first initialization method: {e}")
    logger.error(f"First exception type: {type(e).__name__}")
    
    # Try second initialization method
    try:
        logger.info("Attempting second initialization method with OpenAI class...")
        from openai import OpenAI
        
        # Create a clean configuration with no proxies
        logger.info("Creating OpenAI client with explicit no-proxy configuration...")
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://api.openai.com/v1",  # Use default API endpoint
            timeout=60.0,  # Use default timeout
            max_retries=2  # Use reasonable retry count
            # No proxies parameter
        )
        
        # Test with a simple model list
        models = client.models.list()
        logger.info(f"Second method successful. Available models: {[m.id for m in models][:3]}")
    except Exception as e2:
        logger.error(f"Second initialization method also failed: {e2}")
        logger.error(f"Second exception type: {type(e2).__name__}")
        
        # Final fallback to environment variable approach
        try:
            logger.info("Attempting final initialization method via environment...")
            # Set environment variables directly
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            
            # Attempt to import without the OpenAI module cache
            logger.info("Reloading OpenAI module to clear any cached settings...")
            import sys
            if 'openai' in sys.modules:
                del sys.modules['openai']
            
            # Re-import fresh copy
            from openai import OpenAI
            
            # Create custom configuration for HTTP client
            import httpx
            transport = httpx.HTTPTransport(local_address="0.0.0.0")
            http_client = httpx.Client(transport=transport)
            
            client = OpenAI(
                api_key=OPENAI_API_KEY,
                http_client=http_client
            )
            
            # Test connection
            logger.info("Testing final initialization method...")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            logger.info(f"Final method successful: {response.choices[0].message.content}")
        except Exception as e3:
            logger.error(f"All initialization methods failed. Final error: {e3}")
            logger.error(f"Final exception type: {type(e3).__name__}")
            client = None

# Initialize Flask App
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()  # Use system temp directory

# Add additional logging setup for production
if os.environ.get('FLASK_ENV') != 'development':
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.ERROR)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.ERROR)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'rtf'}

def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Helper Functions ---

def fetch_article_text(url):
    """Fetches HTML from URL and extracts main text content."""
    try:
        headers = { # Pretend to be a browser
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15) # Added timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.content, 'html.parser')

        # --- Basic Content Extraction Logic ---
        # This is simple and might need improvement for complex sites.
        # It finds all paragraph tags and joins their text.
        # Consider libraries like 'newspaper3k' for more robust extraction later.
        paragraphs = soup.find_all('p')
        if not paragraphs:
             # Try finding common article containers if no <p> tags work well
            article_body = soup.find('article') or soup.find('main') or soup.find(class_='content') or soup.find(id='content')
            if article_body:
                 paragraphs = article_body.find_all('p')

        article_text = '\n'.join([p.get_text() for p in paragraphs])

        if not article_text.strip():
            logger.warning(f"Could not extract significant text from URL: {url}")
            return None, "Could not extract article text. The page might be structured unusually, empty, or require JavaScript."

        logger.info(f"Successfully fetched and extracted text from: {url}")
        return article_text, None # Return text and no error

    except requests.exceptions.Timeout:
        logger.error(f"Timeout while fetching URL: {url}")
        return None, f"Timeout: The request to {url} took too long."
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return None, f"Could not fetch URL. Error: {e}"
    except Exception as e:
        logger.error(f"Error parsing content from {url}: {e}")
        return None, f"Could not parse content from the URL. Error: {e}"

def get_summary(text, length='medium'):
    """Generates summary using OpenAI API based on the specified length."""
    if not text or not text.strip():
         return None, "Input text is empty."
    if not client: # Check if client initialized correctly
        return None, "OpenAI client not initialized."

    length_instructions = {
        "short": {
            "prompt": "Create 1-3 extremely brief bullet points capturing only the most essential information. Use phrases rather than full sentences. Format as:\n• Key point 1\n• Key point 2\n• Key point 3\nKeep each point under 10 words when possible:",
            "max_tokens": 250
        },
        "medium": {
            "prompt": "Summarize in EXACTLY one continuous paragraph (4-6 sentences). Do not use bullet points:",
            "max_tokens": 350
        },
        "lengthy": {
            "prompt": "Provide a detailed summary in 2-3 continuous paragraphs. Do not use bullet points:",
            "max_tokens": 650
        }
    }

    if length not in length_instructions:
        length = 'medium'  # Default to medium if invalid length
        logger.warning(f"Invalid length '{length}' provided, defaulting to 'medium'")

    try:
        # Summarize the text with specified length
        logger.info(f"Requesting summary with length: {length}")
        length_config = length_instructions[length]
        summary_prompt = f"{length_config['prompt']}\n\n{text}"[:15000]  # Limit input size

        # Increase the max_tokens based on the length of the input text
        # For very long inputs, we need more tokens for the summary
        input_length = len(text)
        token_multiplier = 1.0
        if input_length > 10000:
            token_multiplier = 1.5
        elif input_length > 5000:
            token_multiplier = 1.2
        
        adjusted_max_tokens = int(length_config['max_tokens'] * token_multiplier)
        logger.info(f"Using adjusted token limit: {adjusted_max_tokens} (base: {length_config['max_tokens']}, multiplier: {token_multiplier})")

        # Create a detailed system message that explicitly defines format requirements for each length
        system_message = """You are a helpful assistant that creates precise summaries. Follow these formatting rules exactly:

1. For SHORT summaries: ALWAYS use 1-3 bullet points with the • character, each under 10 words. Never use paragraphs.
2. For MEDIUM summaries: ALWAYS use exactly ONE continuous paragraph (4-6 sentences). NEVER use bullet points.
3. For LENGTHY summaries: ALWAYS use exactly 2-3 continuous paragraphs. NEVER use bullet points.

Never include the word 'Summary:' in your response."""

        try:
            # Use gpt-3.5-turbo for compatibility
            model_to_use = "gpt-3.5-turbo"
            logger.info(f"Using model: {model_to_use}")
            
            # Create messages array for API call
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": summary_prompt}
            ]
            
            # Call the API using the appropriate client method
            logger.info("Calling OpenAI API for summary generation...")
            summary_response = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                max_tokens=adjusted_max_tokens,
                temperature=0.3,  # Lower temperature for more consistent formatting
            )
                
            # Extract content based on response format
            logger.info(f"Summary response type: {type(summary_response)}")
            if hasattr(summary_response.choices[0], 'message'):
                summary = summary_response.choices[0].message.content.strip()
            else:
                # Fallback in case of different response structure
                logger.info("Using alternate extraction method for summary")
                if hasattr(summary_response.choices[0], 'text'):
                    summary = summary_response.choices[0].text.strip()
                else:
                    # Last resort for dictionary-like responses
                    summary = str(summary_response.choices[0]).strip()
                    
            logger.info("Summary received.")
            
            # Clean up any remaining "Summary:" prefix if it somehow appears
            summary = summary.replace("Summary:", "").strip()

            return summary, None

        except Exception as e:
            logger.error(f"Error with summary generation: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            return None, f"Failed to generate summary: {str(e)}"
            
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        return None, f"Failed to get summary from AI. Error: {e}"

def explain_like_five(summary):
    """Takes a summary and explains it in simple terms a 5-year-old would understand."""
    if not summary or not summary.strip():
        return None, "No summary provided to explain."
    if not client:  # Check if client initialized correctly
        return None, "OpenAI client not initialized."
        
    try:
        # Create system message for ELI5
        system_message = """You are a helpful assistant that explains complex topics in very simple language that a 5-year-old child would understand. 
        Use short, simple sentences, everyday words, and avoid jargon or technical terms. 
        Include relatable examples or analogies where helpful. Be warm and friendly but factually accurate."""
        
        eli5_prompt = f"Explain this summary in a way a 5-year-old would understand:\n\n{summary}"
        
        # Create messages array for API call
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": eli5_prompt}
        ]
        
        # Use gpt-3.5-turbo for compatibility
        model_to_use = "gpt-3.5-turbo"
        logger.info(f"Using model: {model_to_use} for ELI5 explanation")
        
        # Call the API using the appropriate client method
        logger.info("Calling OpenAI API for ELI5 explanation...")
        eli5_response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            max_tokens=600,  # Allow enough space for the explanation
            temperature=0.6,  # Slightly higher temperature for more creative explanations
        )
        
        # Extract content based on response format
        logger.info(f"ELI5 response type: {type(eli5_response)}")
        if hasattr(eli5_response.choices[0], 'message'):
            explanation = eli5_response.choices[0].message.content.strip()
        else:
            # Fallback in case of different response structure
            logger.info("Using alternate extraction method for ELI5")
            if hasattr(eli5_response.choices[0], 'text'):
                explanation = eli5_response.choices[0].text.strip()
            else:
                # Last resort for dictionary-like responses
                explanation = str(eli5_response.choices[0]).strip()
                
        logger.info("ELI5 explanation received.")
        return explanation, None
        
    except Exception as e:
        logger.error(f"OpenAI API error in ELI5: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        return None, f"Failed to get ELI5 explanation. Error: {e}"

def extract_text_from_file(file):
    """Extract text from various file types (PDF, DOCX, TXT)."""
    try:
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        # Save the file temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        extracted_text = ""
        
        # Extract text based on file type
        if file_ext == 'txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                extracted_text = f.read()
                
        elif file_ext == 'pdf':
            if not PyPDF2:
                return None, "PDF processing not available. Please install PyPDF2."
            
            try:
                with open(filepath, 'rb') as f:
                    # Try with the newer API first (PyPDF2 3.0+)
                    try:
                        pdf_reader = PyPDF2.PdfReader(f)
                        for page in pdf_reader.pages:
                            extracted_text += page.extract_text() + "\n"
                    except AttributeError:
                        # Fall back to older API if needed
                        pdf_reader = PyPDF2.PdfFileReader(f)
                        for page_num in range(pdf_reader.getNumPages()):
                            page = pdf_reader.getPage(page_num)
                            extracted_text += page.extractText() + "\n"
            except Exception as e:
                logger.error(f"Error processing PDF: {e}")
                return None, f"Error processing PDF: {str(e)}"
                    
        elif file_ext in ['docx', 'doc']:
            if not docx:
                return None, "DOCX processing not available. Please install python-docx."
            
            try:
                doc = docx.Document(filepath)
                extracted_text = "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                logger.error(f"Error processing DOCX: {e}")
                return None, f"Error processing DOCX: {str(e)}"
            
        elif file_ext == 'rtf':
            # For RTF files, a simple approach is to use a basic text extraction
            # A more robust solution would use a proper RTF parser
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Very basic RTF cleanup
                extracted_text = content.replace('\\par', '\n')
                # Remove RTF control sequences (very basic)
                extracted_text = ' '.join([word for word in extracted_text.split() 
                                         if not word.startswith('\\')])
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        if not extracted_text.strip():
            return None, "Could not extract text from the file. The file may be empty or in an unsupported format."
        
        return extracted_text, None
        
    except Exception as e:
        logger.error(f"Error extracting text from file: {e}")
        return None, f"Error processing file: {str(e)}"

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

# Add error handling for 500 errors
@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace
    app.logger.error('Server Error: %s', e)
    return jsonify({
        "error": "An internal server error occurred. Please check the logs for more details.",
        "details": str(e)
    }), 500

@app.route('/summarize', methods=['POST'])
def handle_summarize():
    """API endpoint to handle summarization requests."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    # Check if OpenAI client is available
    if client is None:
        logger.error("API called but OpenAI client is not initialized")
        return jsonify({
            "error": "OpenAI client not initialized. Please check API key and server logs.",
            "help": "This is likely due to an API key issue. Please ensure your OpenAI API key is correctly set in the environment variables."
        }), 500

    data = request.get_json()
    url = data.get('url')
    text_input = data.get('text')
    length = data.get('length', 'medium')  # Default to medium length

    logger.info(f"Summarize request - URL: {'present' if url else 'none'}, "
               f"Text: {'present' if text_input else 'none'}, "
               f"Length: {length}")

    article_text = None
    error = None

    # --- Input Validation & Content Acquisition ---
    if url and text_input:
        return jsonify({"error": "Please provide EITHER a URL OR paste text, not both."}), 400
    elif url:
        logger.info(f"Processing URL: {url}")
        article_text, error = fetch_article_text(url)
    elif text_input:
        logger.info("Processing pasted text.")
        article_text = text_input # Use pasted text directly
    else:
        return jsonify({"error": "Please provide either a URL, paste article text, or upload a file."}), 400

    # --- Summarization ---
    if error: # If fetching/parsing failed
        return jsonify({"error": error}), 400
    if not article_text: # Should be caught by fetcher, but double-check
         return jsonify({"error": "Failed to get article content."}), 500

    summary, error = get_summary(article_text, length)

    if error:
        return jsonify({"error": error}), 500

    return jsonify({"summary": summary})

@app.route('/explain_like_five', methods=['POST'])
def handle_eli5():
    """API endpoint to handle ELI5 (Explain Like I'm 5) requests."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    # Check if OpenAI client is available
    if client is None:
        logger.error("ELI5 API called but OpenAI client is not initialized")
        return jsonify({
            "error": "OpenAI client not initialized. Please check API key and server logs."
        }), 500

    data = request.get_json()
    summary = data.get('summary')
    
    if not summary:
        return jsonify({"error": "No summary provided to explain."}), 400
        
    logger.info("Processing ELI5 request for summary")
    
    explanation, error = explain_like_five(summary)
    
    if error:
        return jsonify({"error": error}), 500
        
    return jsonify({"explanation": explanation})

@app.route('/summarize_file', methods=['POST'])
def handle_file_summarize():
    """API endpoint to handle file upload and summarization."""
    try:
        # Check if OpenAI client is available
        if client is None:
            logger.error("File API called but OpenAI client is not initialized")
            return jsonify({
                "error": "OpenAI client not initialized. Please check API key and server logs.",
                "help": "This is likely due to an API key issue. Please ensure your OpenAI API key is correctly set in the environment variables."
            }), 500
            
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
            
        file = request.files['file']
        
        # Log debugging information
        logger.info(f"Received file upload: {file.filename}, mimetype: {file.mimetype}")
        
        # Check if the file has a name
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        # Check if the file type is allowed
        if not allowed_file(file.filename):
            return jsonify({"error": f"File type not allowed. Please upload one of these formats: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

        # Get length preference from form data
        length = request.form.get('length', 'medium')  # Default to medium length
        
        logger.info(f"Processing file with length: {length}")
        
        # Extract text from the file
        article_text, error = extract_text_from_file(file)
        
        if error:
            logger.error(f"Error extracting text: {error}")
            return jsonify({"error": error}), 400
            
        if not article_text:
            return jsonify({"error": "Failed to extract text from the file."}), 500
        
        # Generate summary
        summary, error = get_summary(article_text, length)
        
        if error:
            logger.error(f"Error generating summary: {error}")
            return jsonify({"error": error}), 500

        return jsonify({"summary": summary})
    
    except Exception as e:
        logger.exception(f"Unexpected error in file upload handler: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Get port from environment variable (for Heroku, Render, etc.)
    port = int(os.environ.get('PORT', 5001))
    
    # In production, debug should be False
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    # Use host='0.0.0.0' to make it accessible on your network/from outside
    app.run(debug=debug, host='0.0.0.0', port=port)