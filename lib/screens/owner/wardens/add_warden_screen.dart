import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../api.dart';
import '../../../theme.dart';
import '../../../widgets/responsive_shell.dart';

class AddWardenScreen extends StatefulWidget {
  final int hostelId;
  final String hostelName;

  const AddWardenScreen({
    super.key,
    required this.hostelId,
    required this.hostelName,
  });

  @override
  State<AddWardenScreen> createState() =>
      _AddWardenScreenState();
}

class _AddWardenScreenState
    extends State<AddWardenScreen> {
  final _formKey =
      GlobalKey<FormState>();

  final _nameCtrl =
      TextEditingController();

  final _emailCtrl =
      TextEditingController();

  final _phoneCtrl =
      TextEditingController();

  final _cnicCtrl =
      TextEditingController();

  final _cityCtrl =
      TextEditingController();

  bool _saving = false;

  String _selectedShift =
      'Morning';

  final List<String> _floors = [
    'Ground Floor',
    '1st Floor',
    '2nd Floor',
    '3rd Floor',
  ];

  final Set<String> _selectedFloors = {
    // These defaults keep the account unrestricted even though the owner no
    // longer has to configure floor-level access in this simplified form.
    'All Floors',
  };

  final Map<String, bool>
      _permissions = {
    'Manage students': true,
    'View complaints': true,
    'Update attendance': true,
    'Post announcements': true,
    'Manage room allocation': true,
    'View student & parent details': true,
    'Assign penalties': true,
    'Update cleaning schedule': true,
    'Chat with parents & students': true,
  };

  @override
  void dispose() {
    _nameCtrl.dispose();
    _emailCtrl.dispose();
    _phoneCtrl.dispose();
    _cnicCtrl.dispose();
    _cityCtrl.dispose();

    super.dispose();
  }

  // ==========================================================================
  // CREATE WARDEN
  // ==========================================================================

  Future<void> _createWarden() async {
    if (_saving) return;

    final valid =
        _formKey.currentState
                ?.validate() ??
            false;

    if (!valid) return;

    if (_selectedFloors.isEmpty) {
      _showMessage(
        'Please assign at least one floor.',
        error: true,
      );
      return;
    }

    setState(() {
      _saving = true;
    });

    try {
      final response =
          await Api().addWarden(
        hostelId: widget.hostelId,

        name:
            _nameCtrl.text.trim(),

        email:
            _emailCtrl.text.trim(),

        phone:
            _phoneCtrl.text.trim(),

        cnic:
            _cnicCtrl.text.trim(),

        city:
            _cityCtrl.text.trim(),

        shift: _selectedShift,

        floors:
            _selectedFloors.toList(),

        permissions:
            Map<String, bool>.from(
          _permissions,
        ),
      );

      if (!mounted) return;

      final warden =
          Map<String, dynamic>.from(
        response['warden']
                as Map? ??
            {},
      );

      final tempPassword =
          response['temp_password']
                  ?.toString() ??
              '';

      final hostel =
          Map<String, dynamic>.from(
        response['hostel']
                as Map? ??
            {},
      );

      final result =
          <String, dynamic>{
        ...warden,

        'hostel_id':
            widget.hostelId,

        'hostel_name':
            hostel['name'] ??
                widget.hostelName,

        'temp_password':
            tempPassword,

        'status':
            'Offline',

        'accountStatus':
            warden[
                    'account_status'] ??
                'Active',

        'lastActive':
            'Never',

        'joined':
            _formattedToday(),
      };

      await _showCredentialsDialog(
        name:
            warden['name']
                    ?.toString() ??
                _nameCtrl.text.trim(),

        email:
            warden['email']
                    ?.toString() ??
                _emailCtrl.text.trim(),

        password:
            tempPassword,
      );

      if (!mounted) return;

      Navigator.pop(
        context,
        result,
      );
    } catch (error) {
      if (!mounted) return;

      String message =
          error.toString();

      message = message.replaceFirst(
        'Exception: ',
        '',
      );

      _showMessage(
        message,
        error: true,
      );
    } finally {
      if (mounted) {
        setState(() {
          _saving = false;
        });
      }
    }
  }

  // ==========================================================================
  // CREDENTIALS DIALOG
  // ==========================================================================

  Future<void>
      _showCredentialsDialog({
    required String name,
    required String email,
    required String password,
  }) {
    return showDialog<void>(
      context: context,
      barrierDismissible: false,
      // Keep the modal inside the app's nested Navigator so that web/desktop
      // layouts respect ResponsiveShell's phone-frame dimensions.
      useRootNavigator: false,
      builder: (dialogContext) {
        return Dialog(
          backgroundColor:
              Colors.transparent,
          insetPadding:
              const EdgeInsets.symmetric(
            horizontal: 22,
          ),
          child: Container(
            padding:
                const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius:
                  BorderRadius.circular(
                28,
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.black
                      .withOpacity(
                    0.12,
                  ),
                  blurRadius: 35,
                  offset:
                      const Offset(
                    0,
                    16,
                  ),
                ),
              ],
            ),
            child: Column(
              mainAxisSize:
                  MainAxisSize.min,
              crossAxisAlignment:
                  CrossAxisAlignment.start,
              children: [
                Center(
                  child: Container(
                    width: 64,
                    height: 64,
                    decoration:
                        BoxDecoration(
                      color: AppTheme.teal
                          .withOpacity(
                        0.10,
                      ),
                      shape:
                          BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons
                          .verified_user_rounded,
                      color:
                          AppTheme.teal,
                      size: 30,
                    ),
                  ),
                ),

                const SizedBox(
                  height: 18,
                ),

                Center(
                  child: Text(
                    'Warden Account Created',
                    textAlign:
                        TextAlign.center,
                    style: GoogleFonts
                        .playfairDisplay(
                      fontSize: 23,
                      fontWeight:
                          FontWeight.w700,
                      color: AppTheme
                          .textPrimary,
                    ),
                  ),
                ),

                const SizedBox(
                  height: 7,
                ),

                Center(
                  child: Text(
                    '$name has been assigned to ${widget.hostelName}.',
                    textAlign:
                        TextAlign.center,
                    style:
                        GoogleFonts.dmSans(
                      fontSize: 12,
                      height: 1.5,
                      color: AppTheme
                          .textMuted,
                    ),
                  ),
                ),

                const SizedBox(
                  height: 24,
                ),

                Text(
                  'LOGIN CREDENTIALS',
                  style:
                      GoogleFonts.dmSans(
                    fontSize: 10,
                    letterSpacing: 1.4,
                    fontWeight:
                        FontWeight.w700,
                    color:
                        AppTheme.teal,
                  ),
                ),

                const SizedBox(
                  height: 10,
                ),

                _credentialBox(
                  label: 'Email / Login ID',
                  value: email,
                  icon:
                      Icons.mail_outline,
                ),

                const SizedBox(
                  height: 10,
                ),

                _credentialBox(
                  label:
                      'Temporary Password',
                  value: password,
                  icon:
                      Icons.password_rounded,
                ),

                const SizedBox(
                  height: 16,
                ),

                Container(
                  padding:
                      const EdgeInsets.all(
                    14,
                  ),
                  decoration:
                      BoxDecoration(
                    color:
                        const Color(
                      0xFFFFF8EA,
                    ),
                    borderRadius:
                        BorderRadius.circular(
                      15,
                    ),
                    border:
                        Border.all(
                      color:
                          const Color(
                        0xFFF2D79D,
                      ),
                    ),
                  ),
                  child: Row(
                    crossAxisAlignment:
                        CrossAxisAlignment
                            .start,
                    children: [
                      const Icon(
                        Icons
                            .info_outline_rounded,
                        color:
                            Color(
                          0xFFA36B16,
                        ),
                        size: 19,
                      ),

                      const SizedBox(
                        width: 10,
                      ),

                      Expanded(
                        child: Text(
                          'Copy these credentials now. The temporary password is not stored as plain text and cannot be viewed again.',
                          style:
                              GoogleFonts
                                  .dmSans(
                            fontSize: 11,
                            height: 1.45,
                            color:
                                const Color(
                              0xFF74511E,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(
                  height: 22,
                ),

                SizedBox(
                  width:
                      double.infinity,
                  height: 50,
                  child:
                      OutlinedButton.icon(
                    onPressed: () async {
                      await Clipboard.setData(
                        ClipboardData(
                          text:
                              'StayBuddy Warden Login\n'
                              'Hostel: ${widget.hostelName}\n'
                              'Email: $email\n'
                              'Temporary Password: $password',
                        ),
                      );

                      if (!mounted) {
                        return;
                      }

                      _showMessage(
                        'Credentials copied.',
                      );
                    },
                    icon:
                        const Icon(
                      Icons.copy_rounded,
                    ),
                    label: const Text(
                      'Copy Credentials',
                    ),
                    style:
                        OutlinedButton
                            .styleFrom(
                      foregroundColor:
                          AppTheme.teal,
                      side:
                          const BorderSide(
                        color:
                            AppTheme.teal,
                      ),
                      shape:
                          RoundedRectangleBorder(
                        borderRadius:
                            BorderRadius
                                .circular(
                          16,
                        ),
                      ),
                    ),
                  ),
                ),

                const SizedBox(
                  height: 10,
                ),

                SizedBox(
                  width:
                      double.infinity,
                  height: 50,
                  child:
                      ElevatedButton(
                    onPressed: () {
                      Navigator.pop(
                        dialogContext,
                      );
                    },
                    style:
                        ElevatedButton
                            .styleFrom(
                      backgroundColor:
                          AppTheme.teal,
                      foregroundColor:
                          Colors.white,
                      elevation: 0,
                      shape:
                          RoundedRectangleBorder(
                        borderRadius:
                            BorderRadius
                                .circular(
                          16,
                        ),
                      ),
                    ),
                    child: Text(
                      'Done',
                      style:
                          GoogleFonts
                              .dmSans(
                        fontWeight:
                            FontWeight
                                .w700,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _credentialBox({
    required String label,
    required String value,
    required IconData icon,
  }) {
    return Container(
      padding:
          const EdgeInsets.symmetric(
        horizontal: 14,
        vertical: 13,
      ),
      decoration: BoxDecoration(
        color:
            const Color(0xFFF6F8F8),
        borderRadius:
            BorderRadius.circular(16),
        border: Border.all(
          color: AppTheme.border,
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 40,
            height: 40,
            decoration:
                BoxDecoration(
              color: AppTheme.teal
                  .withOpacity(
                0.08,
              ),
              borderRadius:
                  BorderRadius.circular(
                12,
              ),
            ),
            child: Icon(
              icon,
              color:
                  AppTheme.teal,
              size: 19,
            ),
          ),

          const SizedBox(
            width: 12,
          ),

          Expanded(
            child: Column(
              crossAxisAlignment:
                  CrossAxisAlignment
                      .start,
              children: [
                Text(
                  label,
                  style:
                      GoogleFonts.dmSans(
                    fontSize: 10.5,
                    color:
                        AppTheme
                            .textMuted,
                  ),
                ),
                const SizedBox(
                  height: 3,
                ),
                SelectableText(
                  value,
                  style:
                      GoogleFonts.dmSans(
                    fontSize: 13,
                    fontWeight:
                        FontWeight
                            .w700,
                    color:
                        AppTheme
                            .textPrimary,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ==========================================================================
  // UI
  // ==========================================================================

  @override
  Widget build(
    BuildContext context,
  ) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor:
            AppTheme.bg,
        body: SafeArea(
          child: Column(
            children: [
              _header(),

              Expanded(
                child:
                    SingleChildScrollView(
                  physics:
                      const BouncingScrollPhysics(),
                  padding:
                      const EdgeInsets
                          .fromLTRB(
                    18,
                    20,
                    18,
                    120,
                  ),
                  child: Form(
                    key: _formKey,
                    child: Column(
                      crossAxisAlignment:
                          CrossAxisAlignment
                              .start,
                      children: [
                        _sectionTitle(
                          'Personal Information',
                          'Enter the Warden\'s official details.',
                        ),

                        const SizedBox(
                          height: 14,
                        ),

                        _field(
                          controller:
                              _nameCtrl,
                          label:
                              'Full Name',
                          hint:
                              'e.g. Fatima Khan',
                          icon:
                              Icons.person_outline_rounded,
                          validator:
                              (value) {
                            if (value ==
                                    null ||
                                value
                                    .trim()
                                    .isEmpty) {
                              return 'Full name is required';
                            }

                            return null;
                          },
                        ),

                        const SizedBox(
                          height: 12,
                        ),

                        _field(
                          controller:
                              _emailCtrl,
                          label:
                              'Email / Login ID',
                          hint:
                              'warden@example.com',
                          icon:
                              Icons.mail_outline_rounded,
                          keyboardType:
                              TextInputType
                                  .emailAddress,
                          validator:
                              (value) {
                            final email =
                                value
                                        ?.trim() ??
                                    '';

                            if (email
                                .isEmpty) {
                              return 'Email is required';
                            }

                            final valid =
                                RegExp(
                              r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$',
                            ).hasMatch(
                              email,
                            );

                            if (!valid) {
                              return 'Enter a valid email';
                            }

                            return null;
                          },
                        ),

                        const SizedBox(
                          height: 12,
                        ),

                        _field(
                          controller:
                              _phoneCtrl,
                          label:
                              'Phone Number',
                          hint:
                              '03XXXXXXXXX',
                          icon:
                              Icons.phone_outlined,
                          keyboardType:
                              TextInputType
                                  .phone,
                          inputFormatters: [
                            FilteringTextInputFormatter
                                .digitsOnly,
                            LengthLimitingTextInputFormatter(
                              11,
                            ),
                          ],
                          validator:
                              (value) {
                            final phone =
                                value
                                        ?.trim() ??
                                    '';

                            if (phone
                                    .length !=
                                11) {
                              return 'Enter an 11 digit phone number';
                            }

                            if (!phone
                                .startsWith(
                              '03',
                            )) {
                              return 'Phone must start with 03';
                            }

                            return null;
                          },
                        ),

                        const SizedBox(
                          height: 12,
                        ),

                        _field(
                          controller:
                              _cnicCtrl,
                          label: 'CNIC',
                          hint:
                              'XXXXXXXXXXXXX',
                          icon:
                              Icons.badge_outlined,
                          keyboardType:
                              TextInputType
                                  .number,
                          inputFormatters: [
                            FilteringTextInputFormatter
                                .digitsOnly,
                            LengthLimitingTextInputFormatter(
                              13,
                            ),
                          ],
                          validator:
                              (value) {
                            if ((value
                                            ?.trim()
                                            .length ??
                                        0) !=
                                    13) {
                              return 'CNIC must contain 13 digits';
                            }

                            return null;
                          },
                        ),

                        const SizedBox(
                          height: 12,
                        ),

                        _field(
                          controller:
                              _cityCtrl,
                          label: 'City',
                          hint:
                              'e.g. Lahore',
                          icon:
                              Icons.location_city_outlined,
                          validator:
                              (value) {
                            if (value ==
                                    null ||
                                value
                                    .trim()
                                    .isEmpty) {
                              return 'City is required';
                            }

                            return null;
                          },
                        ),

                        const SizedBox(
                          height: 28,
                        ),

                        _sectionTitle(
                          'Hostel Assignment',
                          'This Warden will only manage ${widget.hostelName}.',
                        ),

                        const SizedBox(
                          height: 14,
                        ),

                        _hostelCard(),

                        const SizedBox(
                          height: 26,
                        ),

                        Container(
                          padding:
                              const EdgeInsets
                                  .all(
                            15,
                          ),
                          decoration:
                              BoxDecoration(
                            color:
                                AppTheme
                                    .teal
                                    .withOpacity(
                              0.06,
                            ),
                            borderRadius:
                                BorderRadius
                                    .circular(
                              17,
                            ),
                            border:
                                Border.all(
                              color:
                                  AppTheme
                                      .teal
                                      .withOpacity(
                                0.18,
                              ),
                            ),
                          ),
                          child: Row(
                            crossAxisAlignment:
                                CrossAxisAlignment
                                    .start,
                            children: [
                              const Icon(
                                Icons
                                    .lock_outline_rounded,
                                color:
                                    AppTheme
                                        .teal,
                                size:
                                    20,
                              ),

                              const SizedBox(
                                width: 11,
                              ),

                              Expanded(
                                child: Text(
                                  'StayBuddy will automatically generate a secure temporary password. This Warden will only be authorized for ${widget.hostelName}.',
                                  style:
                                      GoogleFonts
                                          .dmSans(
                                    fontSize:
                                        11.5,
                                    height:
                                        1.5,
                                    color:
                                        AppTheme
                                            .textMuted,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),

                        const SizedBox(
                          height: 22,
                        ),

                        SizedBox(
                          width:
                              double.infinity,
                          height: 54,
                          child:
                              ElevatedButton(
                            onPressed:
                                _saving
                                    ? null
                                    : _createWarden,
                            style:
                                ElevatedButton
                                    .styleFrom(
                              backgroundColor:
                                  AppTheme
                                      .teal,
                              foregroundColor:
                                  Colors
                                      .white,
                              disabledBackgroundColor:
                                  AppTheme
                                      .teal
                                      .withOpacity(
                                0.5,
                              ),
                              elevation:
                                  0,
                              shape:
                                  RoundedRectangleBorder(
                                borderRadius:
                                    BorderRadius
                                        .circular(
                                  17,
                                ),
                              ),
                            ),
                            child:
                                _saving
                                    ? const SizedBox(
                                        width:
                                            21,
                                        height:
                                            21,
                                        child:
                                            CircularProgressIndicator(
                                          strokeWidth:
                                              2.2,
                                          color:
                                              Colors.white,
                                        ),
                                      )
                                    : Row(
                                        mainAxisAlignment:
                                            MainAxisAlignment
                                                .center,
                                        children: [
                                          const Icon(
                                            Icons
                                                .person_add_alt_1_rounded,
                                            size:
                                                19,
                                          ),
                                          const SizedBox(
                                            width:
                                                9,
                                          ),
                                          Text(
                                            'Create Warden Account',
                                            style:
                                                GoogleFonts.dmSans(
                                              fontSize:
                                                  13,
                                              fontWeight:
                                                  FontWeight.w700,
                                            ),
                                          ),
                                        ],
                                      ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _header() {
    return Container(
      padding:
          const EdgeInsets.fromLTRB(
        18,
        13,
        18,
        20,
      ),
      decoration:
          const BoxDecoration(
        gradient: LinearGradient(
          begin:
              Alignment.topLeft,
          end:
              Alignment.bottomRight,
          colors: [
            Color(0xFF073F42),
            Color(0xFF0A696D),
            Color(0xFF0A797B),
          ],
        ),
        borderRadius:
            BorderRadius.vertical(
          bottom:
              Radius.circular(
            30,
          ),
        ),
      ),
      child: Column(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              InkWell(
                onTap:
                    _saving
                        ? null
                        : () =>
                            Navigator.pop(
                              context,
                            ),
                borderRadius:
                    BorderRadius.circular(
                  50,
                ),
                child: Container(
                  width: 42,
                  height: 42,
                  decoration:
                      BoxDecoration(
                    color: Colors.white
                        .withOpacity(
                      0.10,
                    ),
                    shape:
                        BoxShape.circle,
                  ),
                  child:
                      const Icon(
                    Icons
                        .arrow_back_rounded,
                    color:
                        Colors.white,
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(
            height: 17,
          ),

          Text(
            'Add Warden',
            style: GoogleFonts
                .playfairDisplay(
              fontSize: 29,
              fontWeight:
                  FontWeight.w700,
              color: Colors.white,
            ),
          ),

          const SizedBox(
            height: 4,
          ),

          Text(
            widget.hostelName,
            style:
                GoogleFonts.dmSans(
              fontSize: 12.5,
              color: Colors.white
                  .withOpacity(
                0.70,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle(
    String title,
    String subtitle,
  ) {
    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style:
              GoogleFonts.dmSans(
            fontSize: 15,
            fontWeight:
                FontWeight.w700,
            color:
                AppTheme.textPrimary,
          ),
        ),
        const SizedBox(
          height: 4,
        ),
        Text(
          subtitle,
          style:
              GoogleFonts.dmSans(
            fontSize: 11.5,
            height: 1.45,
            color:
                AppTheme.textMuted,
          ),
        ),
      ],
    );
  }

  Widget _hostelCard() {
    return Container(
      width: double.infinity,
      padding:
          const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius:
            BorderRadius.circular(18),
        border: Border.all(
          color: AppTheme.border,
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration:
                BoxDecoration(
              color: AppTheme.teal
                  .withOpacity(
                0.08,
              ),
              borderRadius:
                  BorderRadius.circular(
                15,
              ),
            ),
            child: const Icon(
              Icons
                  .apartment_rounded,
              color:
                  AppTheme.teal,
            ),
          ),

          const SizedBox(
            width: 13,
          ),

          Expanded(
            child: Column(
              crossAxisAlignment:
                  CrossAxisAlignment
                      .start,
              children: [
                Text(
                  widget.hostelName,
                  style:
                      GoogleFonts.dmSans(
                    fontSize: 13,
                    fontWeight:
                        FontWeight
                            .w700,
                    color:
                        AppTheme
                            .textPrimary,
                  ),
                ),

                const SizedBox(
                  height: 3,
                ),

                Text(
                  'Warden access is restricted to this hostel',
                  style:
                      GoogleFonts.dmSans(
                    fontSize: 10.5,
                    color:
                        AppTheme
                            .textMuted,
                  ),
                ),
              ],
            ),
          ),

          const Icon(
            Icons
                .verified_user_outlined,
            color: AppTheme.teal,
            size: 20,
          ),
        ],
      ),
    );
  }

  Widget _shiftSelector() {
    return Row(
      children: [
        'Morning',
        'Evening',
        'Night',
      ].map(
        (shift) {
          final selected =
              _selectedShift ==
                  shift;

          return Expanded(
            child: Padding(
              padding:
                  EdgeInsets.only(
                right:
                    shift != 'Night'
                        ? 8
                        : 0,
              ),
              child: InkWell(
                onTap: () {
                  setState(() {
                    _selectedShift =
                        shift;
                  });
                },
                borderRadius:
                    BorderRadius.circular(
                  14,
                ),
                child: AnimatedContainer(
                  duration:
                      const Duration(
                    milliseconds: 180,
                  ),
                  padding:
                      const EdgeInsets
                          .symmetric(
                    vertical: 13,
                  ),
                  decoration:
                      BoxDecoration(
                    color:
                        selected
                            ? AppTheme
                                .teal
                            : AppTheme
                                .card,
                    borderRadius:
                        BorderRadius
                            .circular(
                      14,
                    ),
                    border:
                        Border.all(
                      color:
                          selected
                              ? AppTheme
                                  .teal
                              : AppTheme
                                  .border,
                    ),
                  ),
                  child: Center(
                    child: Text(
                      shift,
                      style:
                          GoogleFonts
                              .dmSans(
                        fontSize: 11,
                        fontWeight:
                            FontWeight
                                .w700,
                        color:
                            selected
                                ? Colors
                                    .white
                                : AppTheme
                                    .textPrimary,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          );
        },
      ).toList(),
    );
  }

  Widget _permissionsCard() {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius:
            BorderRadius.circular(20),
        border: Border.all(
          color: AppTheme.border,
        ),
      ),
      child: Column(
        children:
            _permissions.keys
                .map(
          (permission) {
            final last =
                permission ==
                    _permissions
                        .keys.last;

            return Column(
              children: [
                SwitchListTile.adaptive(
                  value:
                      _permissions[
                              permission] ??
                          false,
                  onChanged:
                      (value) {
                    setState(() {
                      _permissions[
                              permission] =
                          value;
                    });
                  },
                  activeColor:
                      AppTheme.teal,
                  contentPadding:
                      const EdgeInsets
                          .symmetric(
                    horizontal: 15,
                    vertical: 1,
                  ),
                  title: Text(
                    permission,
                    style:
                        GoogleFonts
                            .dmSans(
                      fontSize: 11.5,
                      fontWeight:
                          FontWeight
                              .w600,
                      color:
                          AppTheme
                              .textPrimary,
                    ),
                  ),
                ),
                if (!last)
                  Divider(
                    height: 1,
                    indent: 15,
                    endIndent: 15,
                    color:
                        AppTheme.border,
                  ),
              ],
            );
          },
        ).toList(),
      ),
    );
  }

  Widget _field({
    required TextEditingController
        controller,
    required String label,
    required String hint,
    required IconData icon,
    String? Function(String?)?
        validator,
    TextInputType?
        keyboardType,
    List<TextInputFormatter>?
        inputFormatters,
  }) {
    return TextFormField(
      controller: controller,
      validator: validator,
      keyboardType:
          keyboardType,
      inputFormatters:
          inputFormatters,
      textInputAction:
          TextInputAction.next,
      style: GoogleFonts.dmSans(
        fontSize: 12.5,
        color:
            AppTheme.textPrimary,
      ),
      decoration:
          InputDecoration(
        labelText: label,
        hintText: hint,
        prefixIcon:
            Icon(
          icon,
          size: 19,
        ),
        filled: true,
        fillColor:
            AppTheme.card,
        labelStyle:
            GoogleFonts.dmSans(
          fontSize: 11.5,
          color:
              AppTheme.textMuted,
        ),
        hintStyle:
            GoogleFonts.dmSans(
          fontSize: 11.5,
          color:
              AppTheme.textMuted
                  .withOpacity(
            0.75,
          ),
        ),
        border:
            OutlineInputBorder(
          borderRadius:
              BorderRadius.circular(
            16,
          ),
          borderSide:
              BorderSide(
            color:
                AppTheme.border,
          ),
        ),
        enabledBorder:
            OutlineInputBorder(
          borderRadius:
              BorderRadius.circular(
            16,
          ),
          borderSide:
              BorderSide(
            color:
                AppTheme.border,
          ),
        ),
        focusedBorder:
            OutlineInputBorder(
          borderRadius:
              BorderRadius.circular(
            16,
          ),
          borderSide:
              const BorderSide(
            color:
                AppTheme.teal,
            width: 1.4,
          ),
        ),
        errorBorder:
            OutlineInputBorder(
          borderRadius:
              BorderRadius.circular(
            16,
          ),
          borderSide:
              const BorderSide(
            color:
                Colors.redAccent,
          ),
        ),
      ),
    );
  }

  String _formattedToday() {
    final now =
        DateTime.now();

    const months = [
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec',
    ];

    return '${now.day.toString().padLeft(2, '0')} '
        '${months[now.month - 1]} '
        '${now.year}';
  }

  void _showMessage(
    String message, {
    bool error = false,
  }) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior:
              SnackBarBehavior
                  .floating,
          backgroundColor:
              error
                  ? const Color(
                      0xFF9D3131,
                    )
                  : const Color(
                      0xFF123F40,
                    ),
          shape:
              RoundedRectangleBorder(
            borderRadius:
                BorderRadius.circular(
              15,
            ),
          ),
          content: Text(
            message,
            style:
                GoogleFonts.dmSans(
              color: Colors.white,
            ),
          ),
        ),
      );
  }
}
