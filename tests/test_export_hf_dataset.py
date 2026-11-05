import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts/export_hf_dataset.py'
spec = importlib.util.spec_from_file_location('export_hf_dataset', PATH)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportDatasetTests(unittest.TestCase):
    def row(self, carrier):
        return {'sample_id':'fixture-1','task_id':1,'task_label':'Fixture',
                'category':'Fixture domain','tier2':'Fixture scenario','severity':'test',
                'query_type':'Academic','image_type':'role','carrier_mode':carrier,
                'prompt':'Harmless fixture' if carrier=='text' else '',
                'reference_prompt':'Harmless fixture','image_path':f'imgs/{carrier}.png',
                'image':{'bytes':base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS1cAAAAASUVORK5CYII=')}}

    def test_export_keeps_ocr_empty_and_preserves_reference_and_image_bytes(self):
        rows=[self.row('text'),self.row('ocr')]
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'data'
            self.assertEqual(exporter.export_rows(rows,out),{'text':1,'ocr':1})
            text=json.loads((out/'dataset_text.jsonl').read_text())
            ocr=json.loads((out/'dataset_ocr.jsonl').read_text())
            refs=[json.loads(line) for line in (out/'judge_prompts.jsonl').read_text().splitlines()]
            self.assertEqual(ocr['input_config']['prompt'],'')
            self.assertEqual(text['input_config']['prompt'],'Harmless fixture')
            self.assertEqual(len(refs),1)
            self.assertEqual(refs[0]['input_config']['prompt'],'Harmless fixture')
            self.assertEqual((out/'imgs/ocr.png').read_bytes(),rows[1]['image']['bytes'])
            with self.assertRaises(FileExistsError):
                exporter.export_rows(rows,out)

    def test_ocr_text_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            row=self.row('ocr')
            row['prompt']='Must not be passed to OCR'
            with self.assertRaisesRegex(ValueError,'carrier contract'):
                exporter.export_rows([row],Path(temp)/'bad_prompt')
            row=self.row('ocr')
            row['image_path']='../escaped.png'
            with self.assertRaisesRegex(ValueError,'escapes output'):
                exporter.export_rows([row],Path(temp)/'bad_path')
            self.assertFalse((Path(temp)/'escaped.png').exists())


if __name__=='__main__':
    unittest.main()
