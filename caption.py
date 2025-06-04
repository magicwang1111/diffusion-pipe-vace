#!/usr/bin/env python
# -*- coding: utf-8 -*-

import google.generativeai as genai
import os
import json
import time
import sys
import traceback # Import traceback for detailed error logging

# ============================================
# <<< --- EDIT THESE CONFIGURATION VALUES --- >>>
# ============================================

# 1. Your Google API Key
# IMPORTANT: Replace "YOUR_API_KEY" with your actual key!
API_KEY = "AIzaSyA0AkROYn7pues_cVFKPa_9BoiAkHY3lHI"

# 2. Path to the FOLDER containing video files to annotate
# IMPORTANT: Replace with the correct path to YOUR folder containing videos!
INPUT_FOLDER_PATH = "/home/wangxi/movie/20250403美妆处理后" # Your example folder path

# 3. The prompt/question to ask the AI about the video (used for all videos)
ANNOTATION_PROMPT = "Describe this video briefly and clearly in 2–3 sentences. Start directly without phrases like 'here is a description'. Include details on appearance, background, clothing, and actions."

# 4. The Gemini model to use (must support video input)
MODEL_NAME = "gemini-2.5-pro-exp-03-25" # Or "gemini-1.5-flash-latest"

# 5. List of video file extensions to look for (lowercase)
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpeg', '.mpg']

# --- NEW CONFIGURATION ---
# 6. Trigger word/phrase to add at the beginning of the output TXT file.
#    Set to "" (empty string) to disable this feature.
TXT_TRIGGER_WORD = "0408美妆," # <<--- SET YOUR DESIRED TRIGGER WORD HERE (e.g., "fashion_video,")

# 7. Timeout for the API call generating content (in seconds)
API_TIMEOUT = 900 # 15 minutes

# 8. Polling interval (seconds) to check file status
FILE_STATUS_POLLING_INTERVAL = 15 # Check every 15 seconds

# 9. Maximum time (seconds) to wait for the file to become ACTIVE
MAX_FILE_PROCESSING_WAIT = 600 # Wait up to 10 minutes

# ============================================
# <<< --- END OF CONFIGURATION SECTION --- >>>
# ============================================


# --- Helper Functions for JSON Serialization ---
# (These functions remain unchanged)
def message_to_dict(message):
    if not message: return None
    result = {}
    if hasattr(message, 'ListFields'):
        for field, value in message.ListFields():
            field_name = field.name
            if hasattr(value, 'ListFields') or isinstance(value, genai.types.MessageType): result[field_name] = message_to_dict(value)
            elif isinstance(value, list) or hasattr(value, 'append'): result[field_name] = [ message_to_dict(item) if hasattr(item, 'ListFields') or isinstance(item, genai.types.MessageType) else item for item in value ]
            else: result[field_name] = value
        return result
    else:
        common_attrs = ['category', 'probability', 'blocked', 'safety_ratings', 'citation_metadata', 'token_count', 'total_token_count', 'prompt_token_count', 'candidates_token_count']
        for attr in common_attrs:
            if hasattr(message, attr):
                value = getattr(message, attr)
                if isinstance(value, list): result[attr] = [message_to_dict(item) if hasattr(item, 'ListFields') or isinstance(item, genai.types.MessageType) else item for item in value]
                elif hasattr(value, 'ListFields') or isinstance(value, genai.types.MessageType): result[attr] = message_to_dict(value)
                else: result[attr] = value
        if hasattr(message, 'text') and 'text' not in result: result['text'] = message.text
        if hasattr(message, 'parts') and 'parts' not in result:
             result['parts'] = []
             for part in message.parts:
                 part_dict = {}
                 if hasattr(part, 'text'): part_dict['text'] = part.text
                 result['parts'].append(part_dict)
    return result if result else str(message)

def candidate_to_dict(candidate):
    if not candidate: return None
    candidate_dict = {}
    if hasattr(candidate, 'content'): candidate_dict['content'] = message_to_dict(candidate.content)
    if hasattr(candidate, 'finish_reason'): candidate_dict['finish_reason'] = str(candidate.finish_reason) if hasattr(candidate.finish_reason, 'name') else candidate.finish_reason
    if hasattr(candidate, 'safety_ratings'): candidate_dict['safety_ratings'] = [message_to_dict(sr) for sr in candidate.safety_ratings]
    if hasattr(candidate, 'citation_metadata'): candidate_dict['citation_metadata'] = message_to_dict(candidate.citation_metadata)
    if hasattr(candidate, 'token_count'): candidate_dict['token_count'] = candidate.token_count
    if hasattr(candidate, 'index'): candidate_dict['index'] = candidate.index
    return candidate_dict

# --- Main Annotation Logic for a SINGLE video ---
# (This function remains unchanged)
def annotate_video(api_key, video_path, prompt_text, model_name, timeout, polling_interval, max_wait):
    """
    Annotates a SINGLE video file using the Google Generative AI API,
    waiting for the file to become ACTIVE.
    (Handles upload, wait, generation, cleanup)
    Returns: (text_response, full_response_object) or (None, None) on failure.
    """
    print(f"--- Starting annotation for: {os.path.basename(video_path)} ---")
    uploaded_file_object = None

    try:
        genai.configure(api_key=api_key)
        print(f"Uploading: {os.path.basename(video_path)}...")
        start_upload_time = time.time()
        uploaded_file_object = genai.upload_file(path=video_path, display_name=os.path.basename(video_path))
        upload_duration = time.time() - start_upload_time
        print(f"Uploaded in {upload_duration:.2f}s. File ID: {uploaded_file_object.name}, Initial State: {uploaded_file_object.state.name}")

        print(f"Waiting for file '{uploaded_file_object.name}' to become ACTIVE...")
        wait_start_time = time.time()
        while uploaded_file_object.state.name == "PROCESSING":
            current_wait_time = time.time() - wait_start_time
            if current_wait_time > max_wait:
                 raise TimeoutError(f"File processing timed out after {max_wait}s for {uploaded_file_object.name}")

            print(f"  State is PROCESSING. Waiting {polling_interval}s... (Elapsed: {current_wait_time:.0f}s / {max_wait}s)")
            time.sleep(polling_interval)
            try:
                uploaded_file_object = genai.get_file(name=uploaded_file_object.name)
            except Exception as get_file_err:
                print(f"\nError refreshing file state for {uploaded_file_object.name}: {get_file_err}", file=sys.stderr)
                raise

        if uploaded_file_object.state.name != "ACTIVE":
            raise ValueError(f"File {uploaded_file_object.name} did not become ACTIVE. Final state: {uploaded_file_object.state.name}.")
        print(f"File {uploaded_file_object.name} is ACTIVE.")

        print(f"Generating content using model: {model_name}...")
        model = genai.GenerativeModel(model_name)
        start_generation_time = time.time()
        contents = [prompt_text, uploaded_file_object]
        response = model.generate_content(contents, request_options={"timeout": timeout})
        generation_duration = time.time() - start_generation_time
        print(f"Content generated in {generation_duration:.2f} seconds.")

        text_response = ""
        try:
            # Extraction logic (same as before)
            if response.candidates:
                for candidate in response.candidates:
                    finish_reason_name = getattr(getattr(candidate, 'finish_reason', None), 'name', "UNKNOWN")
                    if finish_reason_name not in ["STOP", "MAX_TOKENS"]: print(f"  Note: Candidate Finish Reason: {finish_reason_name}")
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'): text_response += part.text
            if not text_response and hasattr(response, 'text'): text_response = response.text
            if not text_response and hasattr(response, 'prompt_feedback'):
                 if hasattr(response.prompt_feedback, 'block_reason'):
                      block_reason_name = getattr(response.prompt_feedback.block_reason, 'name', "UNKNOWN")
                      print(f"Warning: Response blocked. Reason: {block_reason_name}", file=sys.stderr)
                      text_response = f"[Content Generation Blocked - Reason: {block_reason_name}]"
            if not text_response:
                 print("Warning: Could not extract any text content.", file=sys.stderr)
                 text_response = "[No text content generated or extracted]"
        except Exception as e:
             print(f"\nWarning: Error during text extraction: {e}", file=sys.stderr)
             text_response = f"[Error extracting text: {e}]"

        return text_response, response

    except Exception as e:
        print(f"\nAn critical error occurred during annotation for {os.path.basename(video_path)}: {e}", file=sys.stderr)
        # traceback.print_exc() # Uncomment for full traceback during debugging
        return None, None

    finally:
        if uploaded_file_object and hasattr(uploaded_file_object, 'name'):
            file_name_to_delete = uploaded_file_object.name
            print(f"Requesting deletion of uploaded file: {file_name_to_delete}...")
            try:
                 genai.delete_file(file_name_to_delete)
            except Exception as delete_err:
                 print(f"Warning: Could not delete file {file_name_to_delete}. Error: {delete_err}", file=sys.stderr)


# --- Saving Results for a SINGLE video ---
# --- MODIFIED to accept and use trigger_word ---
def save_results(video_path, text_content, response_object, prompt_text, model_name, trigger_word):
    """
    Saves the annotation results (.txt, .json) for a single video
    in the same directory as the video. Prepends trigger_word to the TXT file.

    Args:
        video_path (str): Path to the original video file.
        text_content (str): The extracted text description from the AI.
        response_object (genai.types.GenerateContentResponse): The full API response.
        prompt_text (str): The prompt used for the request.
        model_name (str): The model used for the request.
        trigger_word (str): The word/phrase to prepend to the TXT file (can be empty).
    """
    if not video_path:
        print("Error: Cannot save results - video path is missing.", file=sys.stderr)
        return

    base_path = os.path.splitext(video_path)[0]
    txt_path = base_path + ".txt"
    json_path = base_path + ".json"
    print(f"Saving results for {os.path.basename(video_path)}:")
    print(f"  TXT -> {txt_path}")
    print(f"  JSON -> {json_path}")

    # --- Save text content (.txt file) ---
    try:
        # Determine the base content (handle None case)
        base_txt_content = text_content if text_content is not None else "[Error: No text content retrieved]"

        # Prepare the final content string, prepending the trigger word if provided
        final_txt_content = base_txt_content
        if trigger_word: # Check if trigger_word is not empty
            # Add trigger word and a space before the actual content
            final_txt_content = f"{trigger_word} {base_txt_content}"

        # Write the final combined content to the file
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(final_txt_content)
        # print(f"Text saved successfully.")
    except IOError as e:
        print(f"Error saving text file to {txt_path}: {e}", file=sys.stderr)

    # --- Prepare and Save JSON content (remains unchanged) ---
    serializable_response = {}
    if response_object:
        try:
            if hasattr(response_object, 'candidates'): serializable_response["candidates"] = [candidate_to_dict(c) for c in response_object.candidates]
            if hasattr(response_object, 'prompt_feedback'): serializable_response["prompt_feedback"] = message_to_dict(response_object.prompt_feedback)
            if hasattr(response_object, 'usage_metadata'): serializable_response["usage_metadata"] = message_to_dict(response_object.usage_metadata)
            # Add aggregated_text logic (same as before)
            if text_content and (not serializable_response.get("candidates") or not any('content' in c and 'parts' in c.get('content', {}) and any(p.get('text') == text_content for p in c['content']['parts']) for c in serializable_response.get("candidates", []))):
                 serializable_response["aggregated_text"] = text_content
        except Exception as serial_err:
            print(f"Warning: Could not fully serialize response for JSON: {serial_err}", file=sys.stderr)
            if text_content: serializable_response["raw_text_fallback"] = text_content
            serializable_response["serialization_error"] = str(serial_err)
    else:
        serializable_response = {"error": "No response object received (likely due to failure before generation)."}

    dialogue_data = {
        "request": {
             "video_file_path": video_path,
             "prompt_text": prompt_text,
             "model_used": model_name,
             "request_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "settings": {
                 "trigger_word": trigger_word, # Log the trigger word used
                 "api_timeout_seconds": API_TIMEOUT,
                 "file_processing_max_wait_seconds": MAX_FILE_PROCESSING_WAIT,
                 "file_status_polling_interval_seconds": FILE_STATUS_POLLING_INTERVAL }
        }, "response": serializable_response
    }

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dialogue_data, f, indent=4, ensure_ascii=False)
        # print(f"JSON saved successfully.")
    except (IOError, TypeError) as e:
        print(f"Error saving JSON file to {json_path}: {e}", file=sys.stderr)


# --- Main Batch Processing Execution Block ---
# --- MODIFIED to pass TXT_TRIGGER_WORD to save_results ---
def main():
    """
    Main function to find video files in a folder and process them in batch.
    """
    print("===========================================")
    print("--- Google GenAI Video Annotation Script (Batch Mode with Trigger Word) ---")
    print("===========================================")
    print(f"Input Folder: {INPUT_FOLDER_PATH}")
    print(f"Annotation Prompt: \"{ANNOTATION_PROMPT}\"")
    print(f"Model Selected: {MODEL_NAME}")
    print(f"Looking for file extensions: {', '.join(VIDEO_EXTENSIONS)}")
    print(f"TXT Trigger Word: '{TXT_TRIGGER_WORD}'" + (" (Disabled)" if not TXT_TRIGGER_WORD else ""))
    print("-------------------------------------------")

    # --- Basic Validation Checks (same as before) ---
    abort = False
    if not API_KEY or API_KEY == "YOUR_API_KEY":
        print("\nCRITICAL ERROR: API Key is not set. Edit API_KEY.", file=sys.stderr)
        abort = True
    if not INPUT_FOLDER_PATH or not os.path.isdir(INPUT_FOLDER_PATH):
         print(f"\nCRITICAL ERROR: Input folder not found or invalid: '{INPUT_FOLDER_PATH}'. Edit INPUT_FOLDER_PATH.", file=sys.stderr)
         abort = True
    if abort: sys.exit(1)

    # --- Find Video Files (same as before) ---
    print(f"\nScanning folder '{INPUT_FOLDER_PATH}'...")
    video_files_to_process = []
    try:
        for filename in os.listdir(INPUT_FOLDER_PATH):
            _, ext = os.path.splitext(filename)
            if ext.lower() in VIDEO_EXTENSIONS:
                full_path = os.path.join(INPUT_FOLDER_PATH, filename)
                if os.path.isfile(full_path): video_files_to_process.append(full_path)
    except OSError as e:
        print(f"Error reading directory {INPUT_FOLDER_PATH}: {e}", file=sys.stderr)
        sys.exit(1)

    if not video_files_to_process:
        print(f"No video files found in '{INPUT_FOLDER_PATH}'. Exiting.")
        sys.exit(0)
    print(f"Found {len(video_files_to_process)} video file(s) to process.")

    # --- Process Each Video File ---
    overall_start_time = time.time()
    success_count = 0
    failure_count = 0

    for i, current_video_path in enumerate(video_files_to_process):
        print("\n===========================================")
        print(f"Processing file {i+1} of {len(video_files_to_process)}: {os.path.basename(current_video_path)}")
        print("===========================================")
        file_start_time = time.time()

        try:
            text_result, full_response = annotate_video(
                api_key=API_KEY,
                video_path=current_video_path,
                prompt_text=ANNOTATION_PROMPT,
                model_name=MODEL_NAME,
                timeout=API_TIMEOUT,
                polling_interval=FILE_STATUS_POLLING_INTERVAL,
                max_wait=MAX_FILE_PROCESSING_WAIT
            )

            # Call save_results, passing the configured TXT_TRIGGER_WORD
            if full_response is not None or text_result is not None:
                 save_results(
                     video_path=current_video_path,
                     text_content=text_result,
                     response_object=full_response,
                     prompt_text=ANNOTATION_PROMPT,
                     model_name=MODEL_NAME,
                     trigger_word=TXT_TRIGGER_WORD # Pass the trigger word here
                 )
            else:
                 print("Annotation function returned no results. Skipping save.", file=sys.stderr)

            # Determine success/failure (same logic as before)
            if text_result is not None and not text_result.startswith(("[Error", "[Content Generation Blocked", "[No text content")):
                 print(f"\n--- Successfully processed: {os.path.basename(current_video_path)} ---")
                 success_count += 1
            else:
                 print(f"\n--- Failure/Warning processing: {os.path.basename(current_video_path)} ---", file=sys.stderr)
                 failure_count += 1

        except (TimeoutError, ValueError) as file_state_err:
             print(f"\nPROCESSING FAILED for {os.path.basename(current_video_path)}: {file_state_err}", file=sys.stderr)
             failure_count += 1
        except Exception as e:
            print(f"\nCRITICAL UNEXPECTED ERROR processing {os.path.basename(current_video_path)}: {e}", file=sys.stderr)
            traceback.print_exc()
            failure_count += 1

        file_end_time = time.time()
        print(f"Time taken for this file: {file_end_time - file_start_time:.2f} seconds")

    # --- Final Summary (same as before) ---
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    print("\n===========================================")
    print("--- Batch Processing Complete ---")
    print(f"Input Folder: {INPUT_FOLDER_PATH}")
    print(f"Total files processed: {len(video_files_to_process)}")
    print(f"Successes: {success_count}")
    print(f"Failures/Warnings: {failure_count}")
    print(f"Total execution time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print("===========================================")
    if failure_count > 0: print("\nCompleted with errors.", file=sys.stderr); sys.exit(1)
    else: print("\nCompleted successfully."); sys.exit(0)

# --- Run main function ---
if __name__ == "__main__":
    main()