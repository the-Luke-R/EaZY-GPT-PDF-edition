import os
import sys
from dotenv import load_dotenv
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtWidgets import QFileDialog
from PyPDF2 import PdfReader

from api_key import OPENAI_API_KEY
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

load_dotenv()

pdf = None
chunks = None
embeddings = None
knowledge_base = None

def select_file():
    global pdf
    file_path, _ = QFileDialog.getOpenFileName()
    if file_path:
        # Disable the upload button and enable the remove button
        upload_btn.setEnabled(False)
        remove_btn.setEnabled(True)

        # Update the upload label to display the file name
        upload_label.setText(file_path)

        # Update the button styles
        set_button_enabled_style(upload_btn, False)
        set_button_enabled_style(remove_btn, True)

        pdf = file_path

def clear_input():
    input_box.clear()

def clear_output():
    output_box.clear()

def remove_file():
    # TODO: Remove global variables, I learned a lot since then, and I know there are not advised
    global pdf, chunks, embeddings, knowledge_base
    # Enable the upload button and disable the remove button
    upload_btn.setEnabled(True)
    remove_btn.setEnabled(False)

    # Clear the upload label
    upload_label.clear()

    # Reset the variables
    pdf = None
    chunks = None
    embeddings = None
    knowledge_base = None

    # Update the button styles
    set_button_enabled_style(upload_btn, True)
    set_button_enabled_style(remove_btn, False)

def ask_gpt():
    # extract the text
    global pdf, chunks, embeddings, knowledge_base
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        if chunks is None or embeddings is None or knowledge_base is None:
            # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # create embeddings
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        user_question = input_box.text()
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            # Format the response
            response_lines = response.split("\n")
            formatted_response = "- " + response_lines[0]  # First output starts with "- "
            if output_box.toPlainText():
                formatted_response = "\n\n" + formatted_response  # Add two lines of space if there's existing content in the output box
            if len(response_lines) > 1:
                formatted_response += "\n\n" + "\n\n".join(["- " + line for line in response_lines[1:]])  # Add two lines of space between subsequent outputs

            # Append the formatted response to the output box
            output_box.moveCursor(QtGui.QTextCursor.MoveOperation.End)
            output_box.insertPlainText(formatted_response)

def set_button_enabled_style(button, enabled):
    if enabled:
        button.setStyleSheet("background-color: black; color: white; border: 1px solid white;")
    else:
        button.setStyleSheet("background-color: #333333; color: #666666; border: 1px solid #666666;")

def run_app():
    app = QtWidgets.QApplication(sys.argv)
    root = QtWidgets.QWidget()
    root.setWindowTitle("EaZy ChatGPT PDF")
    root.setStyleSheet("background-color: black;")

    button_frame = QtWidgets.QFrame(root)
    button_frame.setStyleSheet("background-color: black;")
    button_frame_layout = QtWidgets.QHBoxLayout(button_frame)

    global upload_btn, remove_btn, upload_label, input_box, output_box

    upload_btn = QtWidgets.QPushButton("Upload", clicked=select_file)
    set_button_enabled_style(upload_btn, True)
    button_frame_layout.addWidget(upload_btn, 0)

    remove_btn = QtWidgets.QPushButton("Remove File", clicked=remove_file)
    set_button_enabled_style(remove_btn, False)
    button_frame_layout.addWidget(remove_btn, 0)

    root_layout = QtWidgets.QVBoxLayout(root)
    root_layout.addWidget(button_frame)

    upload_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
    upload_label.setStyleSheet("color: #F40F02; background-color: black;")
    root_layout.addWidget(upload_label)

    # Create frame for top section
    top_frame = QtWidgets.QFrame()
    top_frame.setStyleSheet("background-color: black;")
    top_frame_layout = QtWidgets.QVBoxLayout(top_frame)

    # Create labels and input box for left column
    left_frame = QtWidgets.QFrame()
    left_frame_layout = QtWidgets.QVBoxLayout(left_frame)

    ask_label = QtWidgets.QLabel("Ask ChatGPT")
    ask_label.setStyleSheet("font: bold 20px; color: #00b3b3; background-color: black;")
    left_frame_layout.addWidget(ask_label, 0)

    input_box = QtWidgets.QLineEdit()  # Changed to QLineEdit for a single-line input
    input_box.setStyleSheet("background-color: black; color: #00b3b3; font: 12pt 'Lucida Console';")
    input_box.setText("Ask your question here after uploading a file")
    left_frame_layout.addWidget(input_box, 0)

    # Create buttons for middle column
    middle_frame = QtWidgets.QFrame()
    middle_frame_layout = QtWidgets.QHBoxLayout(middle_frame)

    clear_input_btn = QtWidgets.QPushButton("Clear Input Field", clicked=clear_input)
    clear_input_btn.setStyleSheet("font: bold 14px; background-color: grey; color: white; border: 1px solid white;")
    middle_frame_layout.addWidget(clear_input_btn, 0)

    ask_btn = QtWidgets.QPushButton("Ask", clicked=ask_gpt)
    ask_btn.setStyleSheet("font: bold 14px; background-color: #00A67E; color: white")
    middle_frame_layout.addWidget(ask_btn, 0)

    clear_output_btn = QtWidgets.QPushButton("Clear Output Field", clicked=clear_output)
    clear_output_btn.setStyleSheet("font: bold 14px; background-color: grey; color: white; border: 1px solid white;")
    middle_frame_layout.addWidget(clear_output_btn, 0)

    # Create labels and output box for right column
    right_frame = QtWidgets.QFrame()
    right_frame_layout = QtWidgets.QVBoxLayout(right_frame)

    answer_label = QtWidgets.QLabel("ChatGPT Answer")
    answer_label.setStyleSheet("font: bold 20px; color: #00A67E; background-color: black;")
    right_frame_layout.addWidget(answer_label, 0)

    output_box = QtWidgets.QTextEdit()
    output_box.setStyleSheet("background-color: black; color: #00A67E; font: 12pt 'Lucida Console';")
    right_frame_layout.addWidget(output_box, 1)  # Adjusted the stretch factor to occupy the lower half of the window

    top_frame_layout.addWidget(left_frame, 0)
    top_frame_layout.addWidget(middle_frame, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)  # Centered the middle frame horizontally
    top_frame_layout.addWidget(answer_label, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)  # Centered the "ChatGPT Answer" label horizontally
    top_frame_layout.addWidget(output_box, 1)  # Adjusted the stretch factor to occupy the remaining space

    root_layout.addWidget(ask_label, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)  # Centered "Ask ChatGPT" label
    root_layout.addWidget(top_frame)

    root.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
