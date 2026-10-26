import unittest

from oss import SandboxMode, create_sandbox


class MontySandboxTests(unittest.TestCase):
    def setUp(self):
        self.sandbox = create_sandbox(SandboxMode.MONTY)

    def test_executes_python(self):
        self.assertEqual(self.sandbox.execute('print(6 * 7)'), {'output': '42'})

    def test_calls_host_function(self):
        self.assertEqual(self.sandbox.execute('math_sqrt(81)'), {'output': '9.0'})

    def test_reports_runtime_error(self):
        result = self.sandbox.execute('1 / 0')
        self.assertIn('Sandbox runtime error:', result['error'])
        self.assertIn('ZeroDivisionError', result['error'])

    def test_reports_parse_error(self):
        result = self.sandbox.execute('if')
        self.assertIn('Sandbox parse error:', result['error'])


class PythonSandboxTests(unittest.TestCase):
    def setUp(self):
        self.sandbox = create_sandbox(SandboxMode.NONE)

    def test_executes_python(self):
        self.assertEqual(self.sandbox.execute('print(6 * 7)'), {'output': '42\n'})

    def test_reports_runtime_error(self):
        self.assertEqual(self.sandbox.execute('raise ValueError("expected")'), {'error': 'expected', 'output': ''})


if __name__ == '__main__':
    unittest.main()
