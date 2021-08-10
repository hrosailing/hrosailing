import unittest
import hrosailing.polardiagram as pol
import hrosailing.processing as pro

from hrosailing.polardiagram.polardiagram import FileReadingException, FileWritingException


class FileReadingTest(unittest.TestCase):

    def test_read_nonexistent_file(self):
        funcs = [pol.from_csv, pol.depickling, pro.read_csv_file, pro.read_nmea_file]
        for i, f in enumerate(funcs):
            with self.subTest(i=i):
                with self.assertRaises(FileReadingException):
                    f("nonexistentfile")


def reading_suite():
    suite = unittest.TestSuite()
    suite.addTests(
        [
            FileReadingTest("test_read_nonexistent_file")
        ]
    )

    return suite


class FileWritingTest(unittest.TestCase):
    pass


def writing_suite():
    suite = unittest.TestSuite()
    suite.addTests(
        [

        ]
    )

    return suite
