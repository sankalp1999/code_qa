import os
import argparse
from pygls.server import LanguageServer
from lsprotocol.types import ReferenceContext, ReferenceParams, TEXT_DOCUMENT_REFERENCES, TextDocumentIdentifier, Position

def get_class_references(server, file_path):
    document_uri = f"file://{file_path}"
    print(document_uri)
    params = ReferenceParams(
        text_document=TextDocumentIdentifier(uri=document_uri),
        position=Position(line=0, character=0),
        context=ReferenceContext(include_declaration=True),
    )
    response = server.lsp.send_request(TEXT_DOCUMENT_REFERENCES, params).result()
    class_references = {}
    for reference in response:
        class_name = reference["uri"].split("/")[-1].split(".")[0]
        if class_name not in class_references:
            class_references[class_name] = []
        class_references[class_name].append(reference)
    return class_references

def analyze_codebase(codebase_path, language_server_cmd, file_extensions):
    server = LanguageServer("example1", "v1.0")
    print(language_server_cmd)
    server.start_tcp('localhost', 8080, language_server_cmd)
    class_references = {}

    for root, dirs, files in os.walk(codebase_path):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                class_refs = get_class_references(server, file_path)
                class_references.update(class_refs)

    server.stop()
    return class_references

def get_language_server_cmd(language):
    if language == "python":
        return ["pyls"], ['.py']
    elif language == "java":
        return ["java", "-jar", "/opt/homebrew/Cellar/jdtls/1.34.0/libexec/plugins/org.eclipse.equinox.launcher_1.6.800.v20240304-1850.jar", "-configuration", "/opt/homebrew/Cellar/jdtls/1.34.0/libexec/config_mac"], ['.java']
    else:
        raise ValueError(f"Unsupported language: {language}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze class references in a codebase.")
    parser.add_argument("codebase_path", help="Path to the codebase directory.")
    parser.add_argument("--language", choices=["python", "java"], required=True, help="Language of the codebase.")
    args = parser.parse_args()

    codebase_path = args.codebase_path
    language = args.language

    try:
        language_server_cmd, file_extensions = get_language_server_cmd(language)
        class_references = analyze_codebase(codebase_path, language_server_cmd, file_extensions)
        print(f"{language.capitalize()} class references:")
        print(class_references)
    except ValueError as e:
        print(str(e))
