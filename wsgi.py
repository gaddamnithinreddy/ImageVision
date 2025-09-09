"""
WSGI config for ImageVision project.

It exposes the WSGI callable as a module-level variable named ``application``.
"""

from image_recognition_project.app import create_app

application = create_app()

if __name__ == "__main__":
    application.run()
