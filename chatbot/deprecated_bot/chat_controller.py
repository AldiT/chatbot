from response import Response_Generator

class Chat_Controller():
    def __init__(self):
        self.state = "Activated"
        self.negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
        self.exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")


    def process_input(self, user_input: list) -> (str, str, str):

        for item in user_input:
            update_id = item['update_id']
            message = item['message']['text']
            sender_id = item['message']['chat']['id']

        return update_id, message, sender_id

    def make_reply(self, message: str) -> str:
        if message in self.exit_commands:
            self.state = "Deactivated"
            return "This was a good talk. See you later, boy."
            
        if self.state == "Activated":
            self.state = "Active"
            return "Hi, I'm a chatbot trained on random dialogs. Would you like to chat with me?"
        else:
            response_generator = Response_Generator(message)
            return response_generator.generate_response()