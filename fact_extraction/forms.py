from django import forms

class StanceForm(forms.Form):
    claim = forms.CharField(label='claim', max_length=200)
    # CHOICES=(
    #     ('evidence','evidence'),
    #     ('without_evidence','without_evidence')
    # )
    # page_sent = forms.MultipleChoiceField(widget=forms.CheckboxSelectMultiple,
    #                                      choices=CHOICES)