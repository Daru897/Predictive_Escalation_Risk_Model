import csv
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from research.train_candidate import example, load, evaluate

class Contract(unittest.TestCase):
    def row(self,index=0):
        at=datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(hours=index*6)
        return {'ticket_id':str(index),'observed_at':at.isoformat(),'created_at':(at-timedelta(hours=30)).isoformat(),'updated_at':(at-timedelta(hours=4)).isoformat(),'observation_end':(at+timedelta(hours=48)).isoformat(),'escalated_at':(at+timedelta(hours=6)).isoformat() if index%2 else '', 'status':'open','priority':'high' if index%2 else 'normal','transfers':'1','reopens':'0','response_delay_hours':'2','sla_due_at':''}
    def test_labels_and_missing(self):
        self.assertEqual(example(self.row(1))[2],1)
        self.assertEqual(example(self.row(2))[2],0)
        row=self.row();row['transfers']='';self.assertEqual(example(row)[1][8],1)
    def test_leakage_and_censoring(self):
        for field,value in [('updated_at','2099-01-01T00:00:00Z'),('escalated_at','2020-01-01T00:00:00Z'),('observation_end','2025-01-01T01:00:00Z'),('transfers','nan')]:
            row=self.row();row[field]=value
            with self.assertRaises(ValueError):example(row)
    def test_temporal_evaluation_smoke_on_synthetic_data(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'fixture.csv';rows=[self.row(i) for i in range(240)]
            with path.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
            train,test,cutoff=load(path)
            self.assertTrue(all(x[0]+timedelta(hours=24)<cutoff for x in train))
            self.assertTrue(all(x[0]>=cutoff for x in test))
            evaluate(path,Path(d)/'output')
            self.assertTrue((Path(d)/'output/evaluation.json').exists())
            rows[1]['ticket_id']=rows[0]['ticket_id']
            with path.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
            with self.assertRaises(ValueError):load(path)
if __name__=='__main__':unittest.main()
