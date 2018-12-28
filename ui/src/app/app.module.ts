import { BrowserModule } from '@angular/platform-browser';
import { polyfill as keyboardEventKeyPolyfill } from 'keyboardevent-key-polyfill';
import { NgModule } from '@angular/core';
import { TextInputAutocompleteModule } from 'angular-text-input-autocomplete';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { MaterialModule } from './material/material.module';

// // Import ng-circle-progress
import { NgCircleProgressModule } from 'ng-circle-progress';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import {NgxChartsModule} from '@swimlane/ngx-charts';
import { GaugeModule } from 'angular-gauge';
import { NgxGaugeModule } from 'ngx-gauge';

keyboardEventKeyPolyfill();

@NgModule({
  declarations: [
    AppComponent
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    FormsModule,
    ReactiveFormsModule,
    AppRoutingModule,
    MaterialModule,
    HttpClientModule,
    TextInputAutocompleteModule,
    NgxChartsModule,
    GaugeModule.forRoot(),
    NgxGaugeModule,
    NgCircleProgressModule.forRoot({
      // set defaults here
      radius: 100,
      outerStrokeWidth: 16,
      innerStrokeWidth: 8,
      outerStrokeColor: "#78C000",
      innerStrokeColor: "#C7E596",
      animationDuration: 300,
    })
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
