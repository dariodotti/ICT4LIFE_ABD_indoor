from communication.communication import Communication
from communication.domain.messageValueObject import MessageVO

if __name__ == '__main__':
    try:
        message_content = MessageVO(title='test', text="prueba envio", uuid_to="d20d7fc0-c0eb-4d49-8551-745bc149594e",
                                priority="HIGH")
        com = Communication(message_content)
        com.send
    except Exception as e:
        print "ERROR "+ e.message
