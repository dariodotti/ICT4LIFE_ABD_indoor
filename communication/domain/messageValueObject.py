class MessageVO:
    def __init__(self, title='Alert', text='', uuid_to='', priority='LOW'):
        """

        :param title: alert title
        :param text:  alert content
        :param uuid_to:  user destination
        :param priority: alert priority possible values HIGH,MEDIUM or LOW, by default LOW
        """
        self.title = title
        self.text = text
        self.uuid_to = uuid_to
        self.priority = priority if priority in ['HIGH', 'MEDIUM', 'LOW'] else 'LOW'
