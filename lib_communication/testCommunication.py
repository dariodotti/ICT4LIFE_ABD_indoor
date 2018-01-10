from communication.communication import Communication
from communication.domain.messageValueObject import MessageVO

if __name__ == '__main__':
    try:
        message_content = MessageVO(title='test', text="prueba envio", uuid_to="7cceb60a-9d3b-41b6-bcbf-9a504ffe9fb2",
                                priority="HIGH")
        com = Communication(message_content)
        com.send
    except Exception as e:
        print "ERROR "+ e.message
