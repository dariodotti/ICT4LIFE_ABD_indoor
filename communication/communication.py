import requests
import os


class Communication:
    URL = "https://ict4life-services.articatelemedicina.com"

    def __init__(self, message_vo):
        self.message_properties = message_vo

    @property
    def send(self):

        login_request = {"userName": str(os.environ["NELIB_USER"]), "password": str(os.environ["NELIB_PASS"]) }

        login_response = requests.post(self.URL + "/artica-uno-login/loginOauth/login/", json=login_request,
                                       headers={"Accept": "application/json", "Content-Type": "application/json"})

        login_response.raise_for_status()
        if login_response.ok:
            login_data = login_response.json()
            uuid_user = login_data['uuidUser']
            access_token = login_data['token']
            self.__send(access_token, uuid_user)

    def __send(self, access_token, uuid_user):
        headers = {"Accept": "application/json", "access_token": access_token, "uuidUser": uuid_user,
                   "Content-type": "application/json"}

        notifications_request = {
            "destinationUser": self.message_properties.uuid_to,
            "notificationsSource": ["MAIL", "PUSH", "INTERNAL"],
            "content": self.message_properties.text,
            "title": self.message_properties.title,
            'notificationType': "COMMUNICATION",
            "inmediate": "1",
            "originUuid": uuid_user,
            "priority": self.message_properties.priority
        }
        notifications_response = requests.post(self.URL + "/artica-uno-mensajeria/admin/communications/",
                                               json=notifications_request, headers=headers)
        notifications_response.raise_for_status()
