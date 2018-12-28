import { Component, OnInit } from '@angular/core';
import { ApiService } from './api.service';
import { VitalSign } from './model/vital-sign';
import { FormGroup, FormBuilder } from '@angular/forms';
import { NgOnChangesFeature } from '@angular/core/src/render3';
import {NgxChartsModule} from '@swimlane/ngx-charts';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  data: any;
  init_data: any;
  patientForm: FormGroup;
  filteredUsers: any;
  score: any = 0;
  view: any[] = [600, 400];
  gaugeType: any = "arch";
  gaugeLabel: any = "Readmission Probability";
  gaugeAppendText: any = "%";
  gaugeForm: number = 15;
  thresholdConfig: any = {
      '0': {color: '#b9dd2a'},
      '35': {color: '#ddd22c'},
      '65': {color: '#dd992b'},
      '70': {color: '#dd4b2a'}
  };

  colorScheme = {
    domain: ['#f43260', '#97f433']
  };

  formatLabel(value: number | null) {
    if (!value) {
      return 0;
    }
    return value;
  }

  constructor(private apiService: ApiService, private formBuilder: FormBuilder) {

    this.init_data = {
      "discharge_disposition_id": "2",
      "admission_source_id": "4",
      "admission_type_id": "3",
      "number_outpatient": "0",
      "number_inpatient": "0",
      "number_emergency": "0",
      "age": "13",
      "time_in_hospital": "1",
      "num_procedures": "0",
      "num_lab_procedures": "5",
      "num_medications": "12",
      "number_diagnoses": "17",
      "race": "AfricanAmerican",
      "gender": "1",
      "max_glu_serum": "280",
      "A1Cresult": "4.5",
      "diag_1": "E10.65",
      "diag_2": "E10.65",
      "diag_3": "J98.5",
      "metformin": false,
      "repaglinide": false,
      "nateglinide": false,
      "chlorpropamide": false,
      "glimepiride": false,
      "acetohexamide": false,
      "glipizide": false,
      "glyburide": false,
      "tolbutamide": false,
      "pioglitazone": false,
      "rosiglitazone": false,
      "acarbose": false,
      "miglitol": false,
      "troglitazone": false,
      "tolazamide": false,
      "insulin": false,
      "glyburide_metformin": false,
      "glipizide_metformin": false,
      "glimepiride_pioglitazone": false,
      "metformin_rosiglitazone": false,
      "metformin_pioglitazone": false
    };

    this.patientForm = formBuilder.group(this.init_data);
  }

  onSelect(event) {
    console.log(event);
  }

  ngOnInit(): void {

    this.apiService.getPrediction(this.init_data).subscribe((res) => {

      console.log('Created a prediction');
      console.log('res', res);
      this.data = res;
      this.score = (+this.data['probability_1'] * 100).toFixed(2);
      // this.score = [
      //   {
      //     "name": "Readmission",
      //     "value": +this.data['probability_1'] * 100
      //   },
      //   {
      //     "name": "No-Readmission",
      //     "value": +this.data['probability_0'] * 100
      //   }
      // ];
    });

    this.patientForm.valueChanges.subscribe((val: VitalSign) => {

      this.apiService.getPrediction(val).subscribe((res) => {

        console.log('Created a prediction');
        console.log('res', res);
        this.data = res;
        this.score = (+this.data['probability_1'] * 100).toFixed(2);
        // this.score = [
        //   {
        //     "name": "Readmission",
        //     "value": +this.data['probability_1'] * 100
        //   },
        //   {
        //     "name": "No-Readmission",
        //     "value": +this.data['probability_0'] * 100
        //   }
        // ];
      });
    });
  }

  displayFn(user: any) {
    if (user) { return user.ICD; }
  }



}
